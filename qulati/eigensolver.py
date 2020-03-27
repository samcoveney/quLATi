import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import eigsh, eigs
from scipy.spatial.distance import cdist
from scipy.special import cotdg
from scipy.sparse import csr_matrix, lil_matrix, diags

import trimesh

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numba


#{{{ fit a 2D plane to a set of 3D points using lstsq
def fitPlaneLSTSQ(XYZ):
    """Fit 2D plane to 3D points using LSTSQ.

       Borrowed from: https://gist.github.com/RustingSword/e22a11e1d391f2ab1f2c
    """

    (rows, cols) = XYZ.shape
    G = np.ones((rows, 3))
    G[:, 0] = XYZ[:, 0]  #X
    G[:, 1] = XYZ[:, 1]  #Y
    Z = XYZ[:, 2]
    (a, b, c),resid,rank,s = np.linalg.lstsq(G, Z, rcond = None)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return (c, normal)
#}}}


#{{{ calculate eigenfunction gradients
@numba.jit(nopython=True)
def gradientEigenvectors(X, Tri, vertsInFace, V):
    """
        N : meshFine[0:N] are vertices of origianl mesh
        
        * first column of vertsInFace is the centroid, because of how I constructed vertsInFace

        SOLVING A CUBIC POLYNOMIAL FIT TO THE EIGENVALUES OF EACH TRIANGLE...

        Make thew A matrix of features
        f(x,y,z) = f0 + ax + by + cz +
                      + dxx + eyy + fzz + gxy + hyz + izx
                      + jxxx + kyyy + lzzz + mxxy + nxxz + oyyz + pyyx + qzzx + rzzy + sxyz 
        df/dx = a + 2dx + gy + iz + 3jxx + 2mxy + 2nxz + pyy + qzz + syz
        df/dy = b + 2ey + gx + hz + 3kyy + mxx + 2oyz + 2pyx + rzz + sxz
        df/dz = c + 2fz + hy + ix + 3lzz + nxx + oyy + 2qzx + 2rzy + sxy

    """

    print("Calculating gradient of eigenfunctions...")

    # FIXME: only need to do for original faces not in the extended mesh... only need to subDivide original faces too?

    gradV = np.empty((Tri.shape[0], 3, V.shape[1]), dtype = np.float64)
    planepoints = np.empty( (3,3) , dtype = np.float64)
    #dfdz = np.zeros((V.shape[1]), dtype = np.float64)
    gradPhi = np.zeros((V.shape[1], 3), dtype = np.float64)

    #for face, vInF in enumerate(vertsInFace):
    for face in range(vertsInFace.shape[0]):

        vInF = vertsInFace[face]

        # eigenfunctions at vertices in this original face
        phi = V[vInF]

        # basis vectors of triangle: centroid and first two verts
        #points = X[[ vInF[i] for i in [0,1,4] ]]
        planepoints[0] = X[ vInF[1] ] # 1
        planepoints[1] = X[ vInF[4] ] # 4
        planepoints[2] = X[ vInF[7] ] # 7

        v21 = planepoints[1] - planepoints[0]
        v64 = planepoints[2] - planepoints[1]

        n = np.cross(v21, v64)
        n = n / np.linalg.norm(n)
        v = v21 / np.linalg.norm(v21)
        u = np.cross(n, v)

        # change of basis matrix
        R = np.hstack((u,v,n)).reshape(-1,3).T
        
        # all the points in the face rotated down into 2D
        points = X[vInF].dot(R)
        #print("points:", points)

        # plot to check
        #fig = plt.figure()
        #ax = Axes3D(fig)
        #ax.scatter(points[:,0], points[:,1], points[:,2], c = phi[:,1], cmap = "jet")
        #plt.show()


        # fit cubic polynomial model to eigenfunction values
        # --------------------------------------------------

        coords = points.T
        #print("coords:\n", coords) # coordinates of each vertex for the subdivided triangle

        # linear terms
        one = np.ones(10) # f0
        x = coords[0] # a
        y = coords[1] # b

        # quadratic terms
        xx = coords[0]*coords[0] # d
        yy = coords[1]*coords[1] # e
        xy = coords[0]*coords[1] # g

        # cubic terms
        xxx = coords[0]*coords[0]*coords[0] # j
        yyy = coords[1]*coords[1]*coords[1] # k
        xxy = coords[0]*coords[0]*coords[1] # m
        yyx = coords[1]*coords[1]*coords[0] # p

        # stack the features together and reshape properly
        #               f0   a  b  c   d   e   f    g    h    i
        A = np.vstack(( one, x, y, xx, yy, xy, xxx, yyy, xxy, yyx)).reshape(-1, 10).T

        coeffs = np.linalg.lstsq(A, phi)[0]
        #(f0, a, b, c, d, e, f, g, h, i) = coeffs
        f0 = coeffs[0]
        a  = coeffs[1]
        b  = coeffs[2]
        c  = coeffs[3]
        d  = coeffs[4]
        e  = coeffs[5]
        f  = coeffs[6]
        g  = coeffs[7]
        h  = coeffs[8]
        i  = coeffs[9]
        #print("coeffs:", coeffs)


        # check that the function is fit at the centroid
        #centroid_f = f0 + a*x[0] + b*y[0] + c*xx[0] + d*yy[0] + e*xy[0] + f*xxx[0] + g*yyy[0] + h*xxy[0] + i*yyx[0]
        #print("Real phi[c]:", phi[0])
        #print("Predict:  ", centroid_f)

        # derivatives for every eigenfunction
        dfdx = a + 2*c*x[0] + e*y[0] + 3*f*xx[0] + 2*h*xy[0] + i*yy[0]
        dfdy = b + 2*d*y[0] + e*x[0] + 3*g*yy[0] + h*xx[0] + 2*i*xy[0]


        # rotate 2D gradients back into 3D
        # --------------------------------

        # PYTHON: both work in python mode
        #gradPhi = np.hstack((dfdx, dfdy, dfdz)).reshape(-1,V.shape[1]).T
        #gradPhi_1 = np.einsum("ij, jk -> ik", gradPhi, np.linalg.inv(R)) 
        #gradPhi = np.inner(gradPhi, np.linalg.inv(R).T)
        #print(gradPhi_1 - gradPhi_2)
        #input()

        # NUMBA
        gradPhi[:,:] = 0.0
        gradPhi[:,0] = dfdx
        gradPhi[:,1] = dfdy
        invRT = np.linalg.inv(R).T
        for ii in range(V.shape[1]):
            gradPhi[ii, :] = invRT.dot(gradPhi[ii])


        # need to stack gradV as [centroids, components, M]
        gradV[face, 0, :] = gradPhi[:,0]
        gradV[face, 1, :] = gradPhi[:,1]
        gradV[face, 2, :] = gradPhi[:,2]


    return gradV

#}}}


#{{{ subDivide each face into 9 triangles
@numba.jit(nopython=True, fastmath=True)
def subDivide(X, Tri, edges, centroids):
    """My custom subdivide function.

       o : original vertices
       # : vertex at face cetre
       + : vertices on edges
    
           o --- + --- + --- o
            \   / \   / \   /
             \ /   \ /   \ /
              + --- # --- +
               \   / \   /
                \ /   \ /
                 + --- +
                  \   /
                   \ /
                    o
    
       Return the entire new mesh, and info for which vertices are which, etc.

    """
 
    # face centroids i.e. #
    # ---------------------
    #print("centroids shape:", centroids.shape)

    # edges pairs i.e. +
    # ------------------
    edgePairs = np.zeros((edges.shape[0],2,3), dtype = np.float64)
    #print("edgePairs.shape:", edgePairs.shape)
    
    mul = np.array([[1./3.], [2./3.]])

    #for a, edge in enumerate(edges):
    for a in range(edges.shape[0]):
        edge = edges[a]

        e = X[edge[1]] - X[edge[0]]  # edge vector  o ------------ o
        ep = X[edge[0]] + e*mul  # two new points   o -- + -- + -- o
        edgePairs[a] = ep

    
    # create new vertex array
    XN = X.shape[0]
    CN = centroids.shape[0]
    #newX = np.vstack([X, centroids, edgePairs.flatten().reshape(-1, 3)])
    newX = np.empty((X.shape[0] + centroids.shape[0] + edgePairs.shape[0]*2, 3), dtype = np.float64)
    newX[0:X.shape[0]] = X
    newX[X.shape[0]:X.shape[0]+centroids.shape[0]] = centroids
    newX[X.shape[0]+centroids.shape[0]:] = edgePairs.flatten().reshape(-1, 3)


    # stitch all the points together
    # ------------------------------
    # original vertices have original IDs
    # centroid vertices have X.shape[0] + centroidIDs
    # edge vertices have X.shape[0] + centroidIDs.shape[0] + edgePairs.flatten().reshape(-1,3)IDs


    print("Stitching together a new mesh...")

    #newTri = []
    #vertsInFace = [] # for each original face that we subdivide, make a note of all vertices in new face

    newTri = np.empty( (Tri.shape[0]*9, 3), dtype = np.int32 )
    vertsInFace = np.empty( (Tri.shape[0], 10), dtype = np.int32 ) # for each original face that we subdivide, make a note of all vertices in new face

    #for idx, tri in enumerate(mesh.faces):
    count = 0
    for idx in range(Tri.shape[0]):
    
        tri = Tri[idx]
    
        # centroid index into newX for this triangle
        cIdx = XN + idx
        #vInF = [cIdx] # NOTE: record first vertex in this new face as the centroid
        vertsInFace[idx, 0] = cIdx # NOTE: record first vertex in this new face as the centroid
        
        # which edges belong to this triangle
        #whichEdges = np.argwhere( np.isin(edges, tri).sum(axis = 1) == 2 ).flatten()
        whichEdges = np.argwhere( np.sum( (edges == tri[0]) + (edges == tri[1]) + (edges == tri[2]) , axis = 1 ) == 2 ).flatten()


        # this gets me the coordinates of + from edgePairs
        #pairs = edgePairs[whichEdges].flatten().reshape(-1,3)

        # indices into newX of all new points on edges
        pair1Idx_a, pair1Idx_b = XN + CN + 2*whichEdges[0], XN + CN + 2*whichEdges[0] + 1
        pair2Idx_a, pair2Idx_b = XN + CN + 2*whichEdges[1], XN + CN + 2*whichEdges[1] + 1
        pair3Idx_a, pair3Idx_b = XN + CN + 2*whichEdges[2], XN + CN + 2*whichEdges[2] + 1
        pairList = np.array([[pair1Idx_a , pair1Idx_b], [pair2Idx_a , pair2Idx_b], [pair3Idx_a , pair3Idx_b]], dtype = np.int32)

        # connect # to +
        #newTri.append( [ cIdx, pair1Idx_a, pair1Idx_b  ] )
        #newTri.append( [ cIdx, pair2Idx_a, pair2Idx_b  ] )
        #newTri.append( [ cIdx, pair3Idx_a, pair3Idx_b  ] )
        newTri[count, :] = [ cIdx, pair1Idx_a, pair1Idx_b  ]; count += 1
        newTri[count, :] = [ cIdx, pair2Idx_a, pair2Idx_b  ]; count += 1
        newTri[count, :] = [ cIdx, pair3Idx_a, pair3Idx_b  ]; count += 1

        # consider the two nearest vertices + to each corner o;
        #for v in tri:
        for t in range(3):

            v = tri[t]

            # check which edges are connected to current vertex v
            #connect = np.isin(edges[whichEdges], v).any(axis = 1)
            connect =  np.sum( edges[whichEdges] == v, axis = 1) == 1

            # so whichPair tells me how to index pairList if pairList had been shaped differently...
            
            #whichPair = np.argmax( (edges[whichEdges][connect] == v).astype(np.int32), axis = 1)
            whichPair = [ np.argmax( (edges[whichEdges][connect][0] == v) ) , np.argmax( (edges[whichEdges][connect][1] == v) ) ]

            # get the index for the correct p in each pair connected to vertex v
            #res = pairList[connect, whichPair]
            res = [ pairList[connect][0, whichPair[0]] , pairList[connect][1, whichPair[1]] ]

            # connect corner o to these found points +
            #newTri.append( [v, res[0], res[1]] )
            newTri[count, :] = [v, res[0], res[1]]; count += 1

            # connect centre # to these found points +
            #newTri.append( [cIdx, res[0], res[1]] )
            newTri[count, :] = [cIdx, res[0], res[1]]; count += 1

            # update list of verts in new subdivided face
            #vInF.extend([v, res[0], res[1]])
            vertsInFace[idx, 3*t+1:3*t+4] = [v, res[0], res[1]]
        
        #print("vInF:", vInF)
        #input()
        #vertsInFace.append(vInF)

    # create new array
    #newTri = np.array(newTri, dtype = np.int32)

    return newX, newTri, vertsInFace
#}}}


#{{{ calculate laplacian matrix
def laplacian_matrix(X, Tri):

        print("  Calculating Laplacian on atrial mesh")

        RAD_TO_DEG = 180.0 / np.pi

        # approximately normalize edge lengths, to help with units in area calculation
        av_edge = X[Tri[:,0:2]]
        av_edge_length = np.linalg.norm(av_edge[:,1,:] - av_edge[:,0,:], axis = 1).mean()
        X = (X - X.mean(axis = 0)) / av_edge_length

        mesh = trimesh.Trimesh(vertices = X, faces = Tri, process = False)

        areas = mesh.area_faces
        angles = mesh.face_angles # angles within each face, ordered same way as vertices are listed in face

        # make the mass matrix
        vertex_faces = mesh.vertex_faces
        MA = np.ma.masked_array(areas[vertex_faces], vertex_faces < 0) # vertex_faces is padded with -1s
        M = MA.sum(axis = 1) / 3.0 # NOTE: this is the Barycentric area, which is a standard approx for the Voronoi cell 

        # fill out Laplacian by loop over faces
        L = lil_matrix((mesh.vertices.shape[0], mesh.vertices.shape[0]))

        for ff, face in enumerate(mesh.faces):

            cot_ang = cotdg(RAD_TO_DEG * angles[ff, 2])
            L[face[0], face[1]] += cot_ang
            L[face[1], face[0]] += cot_ang

            cot_ang = cotdg(RAD_TO_DEG * angles[ff, 0])
            L[face[1], face[2]] += cot_ang
            L[face[2], face[1]] += cot_ang

            cot_ang = cotdg(RAD_TO_DEG * angles[ff, 1])
            L[face[2], face[0]] += cot_ang
            L[face[0], face[2]] += cot_ang

        # set the diagonals as -sum(rows)
        L.setdiag(-L.sum(axis = 1), k = 0)

        # convert to csr
        L = L.tocsr()
        L = L.multiply(-0.5) # multiply by the half factor, and by -1 to form the negative laplacian

        # do not multiply by inverse mass matrix, instead use this mass matrix in the eigensolver routine
        #L = L.multiply((1.0/M)).tocsr() # need tocsr because otherwise it because a coo_matrix

        M = diags(M)

        return L, M
#}}}


#{{{ solve laplacian eigenproblem
def LaplacianEigenpairs(X, Tri, num = 2**8): 

    print("Get {:d} eigenfunctions with smallest eigenvalues".format(num))

    LS, M = laplacian_matrix(X, Tri)

    print( "  Solving eigenvalue problem for Laplacian...")
    # solve Lx = QMx; using eigsh because LS is symmetric
    [Q,V] = eigsh(LS, k = num, sigma = 0, which = "LM", M = M) #, maxiter = 5000)

    Q, V = np.real(Q), np.real(V) # complex part is zero
    Q[Q < 0] = 0.0 # make sure the extremely tiny first Q is not negative

    return Q, V
#}}}


#{{{ extend mesh
def extendMesh(X, Tri, layers, holes):
    """Extend the mesh with layers of new triangles."""

    print("Extending {:d} holes on mesh with {:d} layers of triangles...".format(holes, layers))

    # loop over the proceedure to build up layers
    for num in range(layers):
        print("Layer {:02d}/{:02d}".format(num+1, layers))

        # trimesh
        mesh = trimesh.Trimesh(vertices = X, faces = Tri, process = False)

        # find edges that belong to one face only
        edges = mesh.edges_unique
        unique, counts = np.unique(mesh.faces_unique_edges, return_counts = True)
        args = np.unique(edges[unique[counts == 1]]) # this gives me the vertices in the edge

        #{{{ plot highlighting the edge vertices
        if False: # and num == layers - 1:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:,0], X[:,1], X[:,2], alpha = 0.15)
            ax.scatter(X[args,0], X[args,1], X[args,2], alpha = 0.5, color = "red")
            plt.show()
        #}}}

        # useful edge information 
        which_edges = np.sum( np.isin(edges, args), axis = 1 ) == 2
        good_edges = edges[which_edges]
        edgeList_total = []

        # loop over holes in the mesh
        for hole in range(0,holes):

            # let's try to group the vertices
            while True:
                vert = args[np.random.choice(np.arange(args.shape[0]))]#[0]
                if hole == 0: break

                if vert not in edgeList_total: break

            firstVert = vert  # save firstVert so we now when we get back there

            edgeList = [firstVert]

            while True:

                result = good_edges[ np.any(np.isin(good_edges, vert), axis = 1) , : ]

                # if anti-clockwise triangles, use next vertex along in a looping fashion (0 -> 1, 1 -> 2, 2 -> 0)...
                if len(edgeList) > 1:
                    face_next = result[  np.any( np.isin(result, vert), axis = 1 ) & np.all( np.isin(result, edgeList[-2], invert = True), axis = 1)].flatten() 
                else:
                    face_next = result[  np.any( np.isin(result, vert), axis = 1 ) ][0,:]

                vert_next = face_next[face_next != vert]

                # break if we get back to where we started
                if vert_next == firstVert: break

                edgeList = edgeList + [vert_next[0]]
                vert = vert_next

            edge_X = X[edgeList]


            # calculate vector 'normal' which points away from holes
            c, normal = fitPlaneLSTSQ(edge_X)

            # flip normal vector if it points inwards
            av_X = np.mean(edge_X, axis = 0)
            inward_vector = av_X - np.mean(X, axis = 0)
            if normal.dot(inward_vector) < 0: normal = normal * -1


            # add new layer of triangles to the edge in question
            # --------------------------------------------------

            new_X = np.copy(X)
            new_Tri = np.copy(Tri)

            # firstly connect new vertices to old vertices
            for ii in range(0, len(edgeList)):
                
                next_i = ii + 1 if ii < len(edgeList) - 1 else 0
                i = ii
                
                newPoint = np.mean(X[edgeList][[i, next_i], :], axis = 0)  + (np.tan(60 * np.pi/180) * np.linalg.norm(X[edgeList[next_i]] - X[edgeList[i]])/2) * normal
                new_X = np.vstack([new_X, newPoint])
                new_Tri = np.vstack([new_Tri, np.array([edgeList[i], edgeList[next_i], new_X.shape[0]-1])])

            # secondly connect new vertices to each other
            for e, ii in enumerate(range(X.shape[0], new_X.shape[0])):

                next_i = ii + 1 if ii < new_X.shape[0] - 1 else X.shape[0]
                i = ii

                if next_i == X.shape[0]:
                    ee = edgeList[0]
                else:
                    ee = edgeList[e + 1]

                new_Tri = np.vstack([new_Tri, np.array([i, next_i, ee])])


            # keep track of which vertices we have dealt with already
            edgeList_total = edgeList_total + edgeList


            # set old mesh equal to new mesh
            X = new_X
            Tri = new_Tri


    print("Extended mesh has {:d} vertices and {:d} faces.".format(X.shape[0], Tri.shape[0]))


    # create another mesh, get edges and centroids
    mesh = trimesh.Trimesh(vertices = X, faces = Tri, process = False)
    edges = mesh.edges_unique
    centroids = mesh.triangles_center

    return X, Tri, edges, centroids
#}}}


# main function
def eigensolver(X, Tri, holes, num = 2**8, layers = 10):
    """Solve the Laplacian eigenproblem for the atrial mesh.

       Method:
       1. extend the mesh
       2. subdivide mesh
       3. calculate eigenfunctions
       4. calculate gradients of eigenfunctions at centroids

       Returns:
       Q: eigenvalues
       V: eigenfunctions at original vertices and face centres
       gradV: eigenfunction gradients at face centres
       centroids: of supplied mesh
    
    """

    orig_num_verts = X.shape[0]
    orig_num_faces = Tri.shape[0]

    # 1. extend the mesh
    # ------------------
    X, Tri, edges, centroids = extendMesh(X, Tri, layers, holes)
    extended_num_verts = X.shape[0] # save number of vertices of extended mesh

    # 2. subdivide the mesh
    # ---------------------
    newX, newTri, vertsInFace = subDivide(X, Tri, edges, centroids)
    #newX, newTri = X, Tri # for testing without subdivide

    # 3. solve eigenproblem on the extended and subdivided mesh
    # ---------------------------------------------------------
    Q, V = LaplacianEigenpairs(newX, newTri, num = num)
    
    # 4. get eigenfunction gradients at mesh centroids
    # ------------------------------------------------
    gradV = gradientEigenvectors(newX, newTri, vertsInFace, V)


    # keep V only at original mesh vertices and centroids
    V = np.vstack( [ V[0:orig_num_verts] , V[extended_num_verts:extended_num_verts+orig_num_faces] ] )

    return Q, V[0:orig_num_verts+orig_num_faces], gradV[0:orig_num_faces, :, :], newX[extended_num_verts:extended_num_verts+orig_num_faces]


