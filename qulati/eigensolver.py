import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist

import pymesh

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


#{{{ calculate eigenvalue gradients

def gradientEigenvectors(meshFine, N, V):
    """
        N : meshFine[0:N] are vertices of origianl mesh
    """

    print("Calculating gradient of eigenfunctions...")

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


def subDivide(mesh):
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
 
    X = mesh.vertices

    # face IDs
    mesh.add_attribute("face_index")
    IDs = mesh.get_attribute("face_index").astype(int)

    # face centroids i.e. #
    # ---------------------
    mesh.add_attribute("face_centroid")
    centroids = mesh.get_attribute("face_centroid").reshape(-1,3)

    print("IDs shape:", IDs.shape)
    print("centroids shape:", centroids.shape)


    # edges pairs i.e. +
    # ------------------
    # when I come to need these values, I will need to call them up in anti-clockwise order...

    __, edges = pymesh.mesh_to_graph(mesh)
    edgePairs = np.zeros([edges.shape[0],2,3])
    print("edgePairs.shape:", edgePairs.shape)
    
    mul = np.array([[1./3.], [2./3.]])

    for a, edge in enumerate(edges):

        e = X[edge[1]] - X[edge[0]]  # edge vector  o ------------ o
        ep = X[edge[0]] + e*mul  # two new points   o -- + -- + -- o
        edgePairs[a] = ep

    
    # need to give these new vertices an ID
    # -------------------------------------
    # original vertices have original IDs
    # centroid vertices have X.shape[0] + centroidIDs
    # edge vertices have X.shape[0] + centroidIDs.shape[0] + edgePairs.flatten().reshape(-1,3)IDs
    # NOTE: I cannot make the list on the fly because many vertices are reused...


    # new edge list:
    XN = X.shape[0]
    CN = centroids.shape[0]
    newX = np.vstack([X, centroids, edgePairs.flatten().reshape(-1, 3)])

    #np.savetxt("/tmp/newX.txt", newX)
    #pymesh.save_mesh("/tmp/mesh.stl", mesh, ascii=True);
#    input("[WAIT]")


    # stitch all the points together
    # ------------------------------

    print("Stitching together a new mesh...")

    newTri = []

    for idx, tri in enumerate(mesh.faces):
    
        #print("\n------ {:d} ------".format(idx))
        #print("tri:", tri)


        # centroid for this triangle
        c = centroids[idx]
        #print("c:", c)
        # should also be here:
        cIdx = XN + idx
        cc = newX[cIdx] # NOTE: XN + idx is the index into newX that we are interested in
        #print("cc:", cc)

        # which edges belonging to this triangle
        #whichEdges = np.argwhere( np.isin(edges, tri).sum(axis = 1) == 2 ).flatten()
        #print("edge IDs:", whichEdges)
        #print("edges:", edges[whichEdges])        

        whichEdges = np.argwhere( np.isin(edges, tri).sum(axis = 1) == 2 ).flatten()
        #print("whichEdges:", whichEdges)

        pairs = edgePairs[whichEdges].flatten().reshape(-1,3) # this gets me the coordinates of + from edgePairs
        #print("pairs:", pairs)

        # TODO: indices into newX of edge points, which is what we want
        pair1Idx_a, pair1Idx_b = XN + CN + 2*whichEdges[0], XN + CN + 2*whichEdges[0] + 1
        pair2Idx_a, pair2Idx_b = XN + CN + 2*whichEdges[1], XN + CN + 2*whichEdges[1] + 1
        pair3Idx_a, pair3Idx_b = XN + CN + 2*whichEdges[2], XN + CN + 2*whichEdges[2] + 1


        # reorder the pairs to go around the triangle anti-clockwise
        # doesn't work because the edgepairs are not in any particular order
        #if norm(newX[tri[0]] - newX[pair1Idx_a]) > norm(newX[tri[0]] - newX[pair1Idx_b]):
        #    pair1Idx_a, pair1Idx_b = pair1Idx_b, pair1Idx_a
        #if norm(newX[tri[1]] - newX[pair2Idx_a]) > norm(newX[tri[1]] - newX[pair2Idx_b]):
        #    pair2Idx_a, pair2Idx_b = pair2Idx_b, pair2Idx_a
        #if norm(newX[tri[2]] - newX[pair3Idx_a]) > norm(newX[tri[2]] - newX[pair3Idx_b]):
        #    pair3Idx_a, pair3Idx_b = pair3Idx_b, pair3Idx_a


        pairList = [pair1Idx_a , pair1Idx_b, pair2Idx_a , pair2Idx_b, pair3Idx_a , pair3Idx_b]
        pair1 = newX[ [pair1Idx_a , pair1Idx_b] ]
        pair2 = newX[ [pair2Idx_a , pair2Idx_b] ]
        pair3 = newX[ [pair3Idx_a , pair3Idx_b] ]
        pairs = np.vstack([pair1, pair2, pair3])
        #print("pairs:", pairs)

        # sanity check figure of the retrieved points
        #fig = plt.figure()
        #ax = Axes3D(fig)
        #ax.scatter(newX[tri][:,0], newX[tri][:,1], newX[tri][:,2], c = np.arange(3), cmap = "jet") #, vmin = MIN, vmax = MAX)
        #ax.scatter(cc[0], cc[1], cc[2], color = "black") #, vmin = MIN, vmax = MAX)
        #ax.scatter(pairs[:,0], pairs[:,1], pairs[:,2], c = np.arange(pairs.shape[0]), cmap = "jet")
        #plt.show()


        # TODO: connect the points together properly by updating newTri

        # connect # to +
        newTri.append( [ cIdx, pair1Idx_a, pair1Idx_b  ] )
        newTri.append( [ cIdx, pair2Idx_a, pair2Idx_b  ] )
        newTri.append( [ cIdx, pair3Idx_a, pair3Idx_b  ] )

        # consider the two nearest vertices + to each corner o;
        # FIXME: this does not always becaus might pick points on same edge! Need closest on each edge, then two closest of those
        # FIXME: can probably do the cdist in a single move instead of looping over corner vertices
        for v in tri:

            # FIXME: even this does not quite work, as sometimes connecting edge and opposite edge have the two closest points
            #        * if I new the edge for each pair, I can keep only the two connecting ones

            #print("v:", v)
            #print("edges:", edges[whichEdges])
            #print("in:", np.isin(edges[whichEdges], v).any(axis = 1) )
            #input()
            connect = np.isin(edges[whichEdges], v).any(axis = 1)

            dists = cdist(newX[v][None,:], pairs).reshape(-1,2)
            #print("dists:", dists)
            d = np.array([0,2,4])[connect] + np.argmin(dists, axis = 1)[connect]
            #print("d:", d)
            d = d[ np.argsort(dists.flatten()[d]) ] # FIXME: only need first two
            #print("dists 2:", dists.flatten()[d])

            # connect corner o to these found points +
            newTri.append( [v, pairList[d[0]], pairList[d[1]] ] )
            # connect centre # to these found points +
            newTri.append( [cIdx, pairList[d[0]], pairList[d[1]] ] )


            #near1 = np.argpartition(cdist(newX[v][None,:], newX[pairList]), 2)[0,0:2]
            #print("near1:", near1)
            #near1 = [v] + pairList[near1[0]] + pairList[near1[1]]
            # connect corner o to these found points +
            #newTri.append( [v, pairList[near1[0]], pairList[near1[1]] ] )
            # connect centre # to these found points +
            #newTri.append( [cIdx, pairList[near1[0]], pairList[near1[1]] ] )

        #print("so far...")
        #print(newTri)

    newTri = np.array(newTri, dtype = np.int32)

    #NEWMESH = pymesh.form_mesh(newX, newTri)
    #pymesh.save_mesh("/tmp/meshNew.stl", NEWMESH, ascii=True);
    input("Done!")

    return newX, newTri
        


#{{{ solve eigenvalue problem for mesh
def LaplacianEigenpairs(X, Tri, num = 256, subdiv = 2): 

    print("Get {:d} eigenfunctions with smallest eigenvalues".format(num))

    mesh = pymesh.form_mesh(X, Tri)

    mesh.enable_connectivity(); # enable connectivity data to be accessed
    mesh.add_attribute("vertex_valance")

    N = mesh.vertices.shape[0]

    # custom sub-divide routine
    subDivide(mesh)

    meshFine = pymesh.subdivide(mesh, order = subdiv, method = "simple")
    #print("ori_face_index:", meshFine.get_attribute("ori_face_index"))

    print("  Calculating Laplacian on atrial mesh using direct pyMesh routine")

    assembler = pymesh.Assembler(meshFine);
    LS = assembler.assemble("laplacian");

    print( "  Solving eigenvalue problem for Laplacian on atrial mesh...")

    # using eigsh: because Ls should be symmetric. Shift invert + LM; should find smallest eigenvalues more easily
    [Q,V] = eigsh(LS, k = num, sigma = 0, which = "LM") #, maxiter = 5000)

    Q, V = np.real(Q), np.real(V) # complex part appears to be zero
    Q[Q < 0] = 0.0 # NOTE: these negative values are basically zero...

    gradEigen = gradientEigenvectors(meshFine, mesh.vertices.shape[0], V)

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

    return Q, V, gradEigen

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
def eigensolver(X, Tri, holes, num = 2**8, layers = 10, subdiv = 2):

    original_size = X.shape[0]

    # extend the mesh
    X, Tri = extendMesh(X, Tri, layers, holes)

    Q, V, gradV = LaplacianEigenpairs(X, Tri, num = num, subdiv = subdiv)

    # keep values only for original (non-extended) mesh vertices
    V, gradV = V[0:original_size], gradV[0:original_size,:,:]

    return Q, V, gradV


