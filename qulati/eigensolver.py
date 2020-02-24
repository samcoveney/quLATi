import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist

import pymesh

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numba

import time

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


#{{{ NEW!!! calculate eigenvalue gradients
@numba.jit(nopython=True)
def NEWgradientEigenvectors(X, Tri, vertsInFace, V):
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


        # need to stack gradV as [3, M, centroids]
        gradV[face, 0, :] = gradPhi[:,0]
        gradV[face, 1, :] = gradPhi[:,1]
        gradV[face, 2, :] = gradPhi[:,2]


    return gradV

#}}}


#{{{ calculate eigenvalue gradients

def gradientEigenvectors(X, Tri, vertsInFace, N, V):
    """
        N : meshFine[0:N] are vertices of origianl mesh
        
        * first column of vertsInFace is the centroid, because of how I constructed vertsInFace
    """

    print("Calculating gradient of eigenfunctions...")
    meshFine = pymesh.form_mesh(X, Tri)
    meshFine.enable_connectivity()

    #{{{ create mesh attributes

    # PyMesh object

    # area
    meshFine.add_attribute("face_area")
    areas = meshFine.get_attribute("face_area")

    # normals
    meshFine.add_attribute("face_normal")
    normals = meshFine.get_face_attribute("face_normal")  # N.B. have not used get_attribute()
    
    meshFine.add_attribute("vertex_normal")
    vertexNormals = meshFine.get_attribute("vertex_normal").reshape(-1,3)

    # face IDs
    meshFine.add_attribute("face_index")
    IDs = meshFine.get_attribute("face_index").astype(int)

    # face centroid
    meshFine.add_attribute("face_centroid")
    centroids = meshFine.get_attribute("face_centroid").reshape(-1,3)

    #}}}


    # pre-allocate array for answers
    gradEigen = np.empty([V.shape[1], 3, N])


    # loop over vertices in original mesh
    for vert, x in enumerate(meshFine.vertices[0:N]):
        #print("vert:", vert)

        neighbouring_faces = meshFine.get_vertex_adjacent_faces(vert)
        #print(neighbouring_faces)

        # loop over neighbouring faces
        for count, face in enumerate(neighbouring_faces):

            # 1. get face vertices
            # --------------------
            tri = meshFine.faces[face]

            # 2. get vertex coordinates
            # -------------------------
            vertices = meshFine.vertices[(tri[2], tri[0], tri[1]), :]  # store vertex coords as [vc, va, vb]

            # 3. normalize vectors, then divide by 2*area
            # -------------------------------------------
            B = np.diff(np.vstack([vertices[-1],vertices]), axis = 0)  # gives us [ [c-b, a-c, b-a] ]
            D = B / (2*areas[face])  # do NOT turn edge vectors B into unit vectors

            # 4. cross product with face normals
            # -----------------------------------
            E = np.cross(normals[face], D)  # if minus sign included, arrows point wrong way, so ignore minus

            # 5. calculate gradients of V
            # ---------------------------
            g = V[tri,:]
            v = np.einsum('ki,kj->ij', g, E)
            #print(v[1,:])
            #print(v.shape)

            # original
            #vSum = v if count == 0 else vSum + v

            # new: weight by areas of the triangles involved
            vSum = v*areas[face] if count == 0 else vSum + v*areas[face]
            areasSum = areas[face] if count == 0 else areasSum + areas[face] 


        # 6. combine gradients at face centres back to central vertex
        # -----------------------------------------------------------

        # original 
        #v_mean = vSum / (count+1) # FIXME: may want to weight this average by face area, or distance of face centre to vertex, or... etc.

        # new: weight by areas of the triangles involved
        v_mean = vSum / areasSum 

        #print(v_mean)
        #print("mean:", v_mean[1,:])
        #input()

        gradEigen[:, :, vert] = v_mean

    
    return gradEigen.T

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
    print("centroids shape:", centroids.shape)

    # edges pairs i.e. +
    # ------------------
    edgePairs = np.zeros((edges.shape[0],2,3), dtype = np.float64)
    print("edgePairs.shape:", edgePairs.shape)
    
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


#{{{ solve eigenvalue problem for mesh
def LaplacianEigenpairs(X, Tri, num = 2**8): 

    print("Get {:d} eigenfunctions with smallest eigenvalues".format(num))

    mesh = pymesh.form_mesh(X, Tri)

    mesh.enable_connectivity(); # enable connectivity data to be accessed
    mesh.add_attribute("vertex_valance")

    N = mesh.vertices.shape[0]

    # custom sub-divide routine
    #subDivide(mesh)

    #meshFine = pymesh.subdivide(mesh, order = subdiv, method = "simple")
    #print("ori_face_index:", meshFine.get_attribute("ori_face_index"))

    meshFine = mesh # NOTE: doing this outside of function

    print("  Calculating Laplacian on atrial mesh using direct pyMesh routine")

    assembler = pymesh.Assembler(meshFine);
    LS = assembler.assemble("laplacian");

    print( "  Solving eigenvalue problem for Laplacian on atrial mesh...")

    # using eigsh: because Ls should be symmetric. Shift invert + LM; should find smallest eigenvalues more easily
    [Q,V] = eigsh(LS, k = num, sigma = 0, which = "LM") #, maxiter = 5000)

    Q, V = np.real(Q), np.real(V) # complex part appears to be zero
    Q[Q < 0] = 0.0 # NOTE: these negative values are basically zero...

    # attempt gradient calculation
#    G = assembler.assemble("gradient");
#    print("G.shape:", G.shape)
#    print("G:", G)
#    input()
#
#    test = G * V[:, 1]  # testing for 1st eigenfunction
#    test = test.reshape([int(test.shape[0]/3),3])
#    print("test shape:", test.shape)
#    print(test)
#    input()

    # check if orthogonal..
    #print("checking orthonormal"); check = V.T.dot((V));  imShow(check)

    return Q, V

#}}}


#{{{ extend mesh
def extendMesh(X, Tri, layers, holes):
    """Extend the mesh with layers of new triangles."""

    print("Extending {:d} holes on mesh with {:d} layers of triangles...".format(holes, layers))

    # loop over the proceedure to build up layers
    for num in range(layers):
        print("Layer {:02d}/{:02d}".format(num+1, layers))

        mesh = pymesh.form_mesh(X, Tri)
        #print("mesh:", mesh)
            

        # use Gaussian Curvature to capture the mesh edges
        mesh.add_attribute("vertex_gaussian_curvature")
        curvature = mesh.get_attribute("vertex_gaussian_curvature")
        condition = curvature > 10.0*np.nanmean(curvature)
        #print(condition)
        args = np.argwhere(condition == True)
        #print(args)


        if False:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:,0], X[:,1], X[:,2], alpha = 0.15) #, c = V[:,i])
            ax.scatter(X[condition,0], X[condition,1], X[condition,2], alpha = 0.5, color = "red")
            plt.show()


        # try to use valence, i.e. how many neighbours, to remove bad cases
        if False:
            mesh.add_attribute("vertex_valance")
            valence = mesh.get_attribute("vertex_valance")
            print("valence:", valence)
            condition = valence < 6


        __, edges = pymesh.mesh_to_graph(mesh)
        #print(edges)


        which_edges = np.sum( np.isin(edges, args), axis = 1 ) == 2
        #print(edges[which_edges])
        good_edges = edges[which_edges]

        edgeList_total = []

        # FIXME: hardwired currently for 5 holes in the mesh, needs generalizing
        for hole in range(0,holes):

            #print(edgeList_total)
            #input("wait")

            # let's try to group the vertices
            while True:
                vert = args[np.random.choice(np.arange(args.shape[0]))][0]
                if hole == 0: break

                if vert not in edgeList_total: break


            #print("vert:", vert)
            #input("waiting:")
            firstVert = vert  # save firstVert so we now when we get back there

            edgeList = [firstVert]

            while True:

                result = good_edges[ np.any(np.isin(good_edges, vert), axis = 1) , : ]
                #print("Tri_with_condition:\n", result)

                # if anti-clockwise triangles, use next vertex along in a looping fashion (0 -> 1, 1 -> 2, 2 -> 0)...
                if len(edgeList) > 1:
                    #print(edgeList[-2])
                    #print( np.all( np.isin(result, edgeList[-2], invert = True), axis = 1) )
                    face_next = result[  np.any( np.isin(result, vert), axis = 1 ) & np.all( np.isin(result, edgeList[-2], invert = True), axis = 1)].flatten() 
                else:
                    face_next = result[  np.any( np.isin(result, vert), axis = 1 ) ][0,:]
                #print("face_next:", face_next)

                vert_next = face_next[face_next != vert]


                # break if we get back to where we started
                if vert_next == firstVert: break

                edgeList = edgeList + [vert_next[0]]
                vert = vert_next
                #print("new vert:", vert)

            #print("edge list:", edgeList)

            edge_X = X[edgeList]


            # calculate vector 'normal' which points away from holes
            # ------------------------------------------------------

            if False:
                # compute centered coordinates
                G = edge_X.sum(axis=0) / edge_X.shape[0]
                # run SVD
                u, s, vh = np.linalg.svd(edge_X - G)
                # unitary normal vector
                normal = vh[2, :]
            else:
                c, normal = fitPlaneLSTSQ(edge_X)

            #print(np.linalg.norm(normal))


            av_X = np.mean(edge_X, axis = 0)


            # flip normal vector if it points inwards
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

    return X, Tri
#}}}


# main function
def eigensolver(X, Tri, holes, num = 2**8, layers = 10):
    """Solve the Laplacian eigenproblem for the atrial mesh.

       Method:
       1. extend the mesh
       2. subdivide mesh
       3. calculate eigenfunctions
       4. calculate gradients of eigenfunctions at centroids

       These routine rely on PyMesh, but only for conveniance:
       * centroids: easily calculated as mean of vertics
       * edges: easily contructed from faces
       * "Laplacian": this could be coded up easily, but I haven't done it yet
       * "vertexGaussianCurvature": used to identify edge points. Probably simple to replace 
       
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
    X, Tri = extendMesh(X, Tri, layers, holes)
    extended_num_verts = X.shape[0]

    # 2. subdivide the mesh
    # ---------------------
    # use pymesh routines to get edge pairs and centroids
    mesh = pymesh.form_mesh(X, Tri)
    __, edges = pymesh.mesh_to_graph(mesh)
    mesh.add_attribute("face_centroid")
    centroids = mesh.get_attribute("face_centroid").reshape(-1,3)

    # call to subdivide function
    newX, newTri, vertsInFace = subDivide(X, Tri, edges, centroids)

    # 3. solve eigenproblem on the extended and subdivided mesh
    # ---------------------------------------------------------
    Q, V = LaplacianEigenpairs(newX, newTri, num = num)
    
    # get the gradients at the centroids of the original mesh
    gradV = NEWgradientEigenvectors(newX, newTri, vertsInFace, V)

    # keep values only for original (non-extended) mesh vertices
    # FIXME: which values am I keeping?
    #V, gradV = V[0:original_size], gradV[0:original_size,:,:]

    # return V for original vertices + centroids, gradV was only calculated for centroids anyway

    V = np.vstack( [ V[0:orig_num_verts] , V[extended_num_verts:extended_num_verts+orig_num_faces] ] )

    return Q, V[0:orig_num_verts+orig_num_faces], gradV[0:orig_num_faces, :, :], newX[extended_num_verts:extended_num_verts+orig_num_faces]


