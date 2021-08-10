import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
import random
#from multiprocessing import Process, Queue

import trimesh

import numba


#{{{ fit a 2D plane to a set of 3D points using lstsq
def __fitPlaneLSTSQ(XYZ):
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


#{{{ extend mesh
def extendMesh(X, Tri, layers, holes, use_average_edge = False):
    """ Extend the mesh with layers of new triangles.
        
        layers: how many layers of new elements to add
        holes: how many topological holes (e.g. mitral valve, pulmondary veins)
        use_average_edge: instead of using local edge lengths for mesh extension, use the average edge length for all edges around the holes.
                          This can help to extend meshes with jagged/sawtooth edges (but perhaps it works less well with an irregular triangulation?).
    """

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
            import pyvista as pv
            plotter = pv.Plotter()

            plt_surf = pv.PolyData(X, np.hstack([ np.full(Tri.shape[0], 3)[:,None] , Tri ]))
            plotter.add_mesh(plt_surf, show_edges = True, opacity = 1.0) # add color to mesh here

            plt_points = pv.PolyData(X[args])
            plotter.add_mesh(plt_points, color = "red", point_size = 10, render_points_as_spheres = True) # add color to mesh here

            plotter.show()
        #}}}

        # useful edge information 
        which_edges = np.sum( np.isin(edges[unique[counts == 1]], args), axis = 1 ) == 2 # FIXME: case of an element 'tooth' on edge is capture here, needs discarding
        good_edges = edges[unique[counts == 1]][which_edges]
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
                #print("vert:", vert)
                #print("result:", result)

                # if anti-clockwise triangles, use next vertex along in a looping fashion (0 -> 1, 1 -> 2, 2 -> 0)...
                if len(edgeList) > 1:
                    face_next = result[  np.any( np.isin(result, vert), axis = 1 ) & np.all( np.isin(result, edgeList[-2], invert = True), axis = 1)].flatten() 
                else:
                    face_next = result[  np.any( np.isin(result, vert), axis = 1 ) ][0,:]
                #print("face_next:", face_next)

                vert_next = face_next[face_next != vert]

                if vert_next == firstVert: break

                edgeList = edgeList + [vert_next[0]]
                vert = vert_next

            edge_X = X[edgeList]

            if use_average_edge:
                av_edge_length = np.linalg.norm(edge_X[1:,:] - edge_X[:-1,:], axis = 1).mean()

            # calculate vector 'normal' which points away from holes
            c, normal = __fitPlaneLSTSQ(edge_X)

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
                
                if use_average_edge:
                    newPoint = np.mean(X[edgeList][[i, next_i], :], axis = 0) \
                             + (np.tan(60*np.pi/180) * av_edge_length/2.0) * normal
                else:
                    newPoint = np.mean(X[edgeList][[i, next_i], :], axis = 0) \
                             + (np.tan(60 * np.pi/180) * np.linalg.norm(X[edgeList[next_i]] - X[edgeList[i]])/2) * normal

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


# calculate energy
#def calc_energy(q, X, points, neigh, neigh_ind, choice_ind, POWER):
def calc_energy(X, points, neigh, neigh_ind, choice_ind, POWER):
    """Calculate the energy needed for simulated annealing"""

    #np.seterr(divide='ignore', invalid='ignore')

    # calculate distances between all points and the point in question
    old_dists = cdist(points[choice_ind], points)[0] #  at initial position
    new_dists = cdist(X[neigh[neigh_ind]][None,:], points)[0] # at suggested new location

    # calculate the energy as sum of squared inverse distances
    # (ignore distance between the point and itself (which is zero) when calculating the energy)
    old_energy = (1.0 / old_dists[:choice_ind[0]]**POWER).sum() + (1.0 / old_dists[choice_ind[0]+1:]**POWER).sum()
    new_energy = (1.0 / new_dists[:choice_ind[0]]**POWER).sum() + (1.0 / new_dists[choice_ind[0]+1:]**POWER).sum()

    del old_dists
    del new_dists

    #q.put([old_energy, new_energy])

    return old_energy, new_energy


#{{{ anneal positions of a subset of mesh vertices, optimizing for even spatial distribution
def subset_anneal(X, Tri, num, runs, choice = None):
    """ Use simulated annealing to distribute vertex points over a manifold.
    
        Arguments:
        X: 3D vertex coordinates
        Tri: elements of triangulation
        num: number of points in the subset
        runs: maximum number of runs
        choice: indices of a previously annealed subset
    """

    print("Optimizing inducing point positions with simulated annealing...")

    #np.seterr(divide='ignore', invalid='ignore')

    # trimesh object
    mesh = trimesh.Trimesh(vertices = X, faces = Tri, process = False)

    # firstly, find a spread out set of vertices where we had observations
    num_designs = runs

    # initial design
    if choice is None:
        #choice = np.unique(np.random.choice(X.shape[0], size = num))
        choice = np.arange(X.shape[0])
        np.random.shuffle(choice)
        choice = choice[0:num]
        #print("choice.shape:", choice.shape)
        #print("choice type:", choice.dtype) # <------------------- HERE

    points = X[choice]
    dists = pdist(points)
    POWER = 2 # power to which inverse distance is raised; 2 seems to be a good choice
    best_cost = ((1.0 / dists)**POWER).sum() # sum for energy
    del dists

    # run the simulated annealing routine
    count = 0
    for i in range(1, num_designs + 1):

        # choose a vertex that we are going to try to move around
        choice_ind = np.random.choice(choice.shape[0], size = 1)

        # which vertices neighbour the current point at choice[choice_ind]?
        #neigh = mesh.vertex_neighbors[choice[choice_ind]][0]  # old indexing, when mesh.vertex_neighbours returned an array with a list in it i.e. [list(a,b,c)]
        neigh = mesh.vertex_neighbors[choice[choice_ind][0]]   # new indexing, when mesh.vertex_neighours returns [a,b,c]

        # choose a random neighbour
        neigh_ind = np.random.choice(len(neigh), size = 1)[0]

        # use subprocess to prevent memory usage increasing
#        queue = Queue()
#        p = Process(target = calc_energy, args=(queue, X, points, neigh, neigh_ind, choice_ind, POWER))
#        p.start()
#        p.join() # this blocks until the process terminates
#        [old_energy, new_energy] = queue.get()

        [old_energy, new_energy] = calc_energy(X, points, neigh, neigh_ind, choice_ind, POWER)

        # energy difference
        diff_energy = new_energy - old_energy

        # if using temperature, which does not seem to help much
        #temperature = 1e-5*( 1.0 - (i+1)/num_designs )
        #print("temp:", temperature)
        #prob = np.exp(-diff_energy/temperature)
        #print("prob:", prob)

        if (diff_energy < 0): # or (prob > np.random.uniform()):
            count += 1
            best_cost = best_cost + (new_energy - old_energy) # sum for energy - probably want E = sum(dists^2) to square this energy though...
            #print("new energy:", best_cost)

            # update the best choice
            choice[choice_ind] = neigh[neigh_ind]
            points[:] = X[choice]
        
        if i % 10000 == 0:
            perc = 100*count/10000
            print("Progress {:02d}%, Percentage of successful moves: {:4.1f}%".format(int(100*i/num_designs), perc), end = "\r")
            count = 0
            if perc < 1.0:
                print("\nBreaking at <= 1% successful moves")
                break

    return choice
#}}}


#{{{ triangulate a subset of mesh vertices into a new triangulation
def subset_triangulate(X, Tri, choice, layers = 0, holes = 5, use_average_edge = True):

    original_N, original_F = X.shape[0], Tri.shape[0]

    if layers > 0:
        X, Tri, __, __ = extendMesh(X, Tri, layers = layers, holes = holes, use_average_edge = use_average_edge)
    

    #{{{ for each point in high res mesh, which point X[choice] is nearest?
    print("Calculating nearest inducing point")
    closest_c = np.empty(X.shape[0], dtype = np.int32)

    chunkSize = 10000
    P = X.shape[0]
    print("Calculating closest new vertex for {:d} vertices in high res mesh".format(P))
    if P > chunkSize:
        chunkNum = int(np.ceil(P / chunkSize))
        print("  Using", chunkNum, "chunks of", chunkSize)
    else:
        chunkNum = 1

    ## loop over outputs (i.e. over emulators)
    #printProgBar(0, chunkNum, prefix = '  Progress:', suffix = '')
    for c in range(chunkNum):
        L = c*chunkSize
        U = (c+1)*chunkSize if c < chunkNum -1 else P

        closest_c[L:U] = np.argmin(cdist(X[L:U], X[choice,:]), axis = 1)
        #printProgBar(U+1, P, prefix = '  Progress:', suffix = '')
        #printProgBar(c+1, chunkNum, prefix = '  Progress:', suffix = '')

    #}}}


    #{{{ build an edge point of connected regions
    print("Building edge list...")

    # find edges that belong to one face only
    mesh = trimesh.Trimesh(vertices = X, faces = Tri, process = False)  # NOTE: !!!! WARNING! REMEMBER TO PASS THE RELEVANT PARTS OF 'TRI' !!!!
    edges = mesh.edges_unique # these edges are for the high resolution mesh

    closest_c_edges = closest_c[edges.flatten()].reshape(-1,2) # replace index into high res mesh with index of closest inducing point c

    # check where these edges belond to two different Voronoi regions i.e. keep edges where the connecting regions are different
    condition = ( (closest_c_edges[:,1] - closest_c_edges[:,0]) != 0 )

    # these edges should be between the inducing points
    edge_list = closest_c_edges[condition]

    # keep only the unique edges that connect different regions
    edge_list = np.sort(edge_list, axis = 1)
    edge_list = np.unique(edge_list, axis = 0)

    #}}}

 
    #{{{ make triangulation from edge list
    
    print("Building face list...")
    face_list = []
    for cc, c in enumerate(choice): # loop over inducing points

        which_edges = edge_list[np.any(np.isin(edge_list, cc), axis = 1)]

        tmp = np.unique(which_edges)
        tmp = np.unique(tmp[tmp != cc])

        # trying to find a single edge that contains two of these vertices
        ind = (np.sum(np.isin(edge_list, tmp), axis = 1) == 2)
        
        try:
            tmp_2 = np.unique(edge_list[ind], axis = 1)

            for i in tmp_2:
                lst = list(i)
                lst.append(cc)
                face_list.append(lst)

        except:
            pass
        
    face_list = np.array(face_list, np.int32)

    # make sure faces are not repeated
    face_list = np.sort(face_list, axis = 1)
    face_list = np.unique(face_list, axis = 0)


    # remove faces so that max faces per edge is 2
    # --------------------------------------------
    # loop over this structure until no offending triangles, then break
    while True:
        mesh = trimesh.Trimesh(vertices = X[choice], faces = face_list, process = False)
        edges = mesh.edges_unique
        unique, counts = np.unique(mesh.faces_unique_edges, return_counts = True)

        args = np.unique(edges[unique[(counts > 2)]]) # offending triangles will contain all these vertices
        #print("args:", args)

        if len(args) == 0: break
        
        good_edges = edges[unique[(counts == 2)]]

        # find the good edges that contains the offending vertices
        ttt = good_edges[ np.all(np.isin(good_edges, args), axis = 1) ]

        try:
            # pick the first one, and remove triangles containing it
            defo_bad_face = (np.isin(face_list, ttt[0]).sum(axis = 1) == 2 )
        except:
            # handle the case of single triangles over 3 other triangles
            defo_bad_face = np.all(np.isin(face_list, args), axis = 1)

        #print("bad faces")
        #print(face_list[defo_bad_face])

        face_list = face_list[~defo_bad_face]

    #}}}


    #{{{ delete some poor quality triangles that may have formed on the edge

    trimesh_obj = trimesh.Trimesh(vertices = X[choice], faces = face_list, process = False)
    
    if True:
        # two loops of removing obtuse triangles in the edge should help
        for i in range(2):
            # which faces have large angles
            angles = trimesh_obj.face_angles # angles within each face, ordered same way as vertices are listed in face
            DEG_TO_RAD = np.pi/180.0
            bad_triangles = np.any(angles > 135 * DEG_TO_RAD, axis = 1)

            # which faces border the holes?
            edges = trimesh_obj.edges_unique
            unique, counts = np.unique(trimesh_obj.faces_unique_edges, return_counts = True)

            is_edge_face = np.any(np.isin(trimesh_obj.faces_unique_edges, unique[counts == 1]), axis = 1) # which faces contain an edge that belongs to only one face?

            face_list = face_list[~(bad_triangles & is_edge_face)]

            trimesh_obj = trimesh.Trimesh(vertices = X[choice], faces = face_list, process = False)

    #}}}

    # fix the normals to be consistent
    trimesh_obj.fix_normals()

    # return the faces
    return X[choice], trimesh_obj.faces, closest_c[0:original_N]
#}}}


#{{{ ensure no elements exist that have two edges bordering a hole
def fix_edges(X, Tri):
    """Ensure that no face has two edges bordering a hole, by insertion of additional elements."""

    # trimesh
    mesh = trimesh.Trimesh(vertices = X, faces = Tri, process = False)

    # find edges that belong to one face only
    edges = mesh.edges_unique # list of unique edges
    unique, counts = np.unique(mesh.faces_unique_edges, return_counts = True) # which edges belong to only one face?

    # which unique edges appear only once in the entire face list (i.e. are not shared between two faces?)
    args = np.unique(edges[unique[counts == 1]]) # this gives me the vertices in the edge

    # which face has 3 vertices that are in a unique edge?
    dodgy_face_idx = np.where(np.isin(mesh.faces, args).sum(axis = 1) == 3)[0]
    print("dodgy_face_idx:", dodgy_face_idx)

    # loop over these dodgy faces
    for dfi in dodgy_face_idx:
        dodgy_face = mesh.faces[dfi]
        print("dodgy_face:", dodgy_face)

        # three edges of the dodgy face
        three_edges = mesh.faces_unique_edges[dfi]
        print("three_edges:", three_edges)

        res = np.isin(three_edges, unique[counts == 1])
        print("res:", res)

        bad_edges = edges[three_edges[res]]
        print("bad_edges:", bad_edges)


        # which vertex is in the tip of the element? (i.e. shared by two edges)
        v, c = np.unique(bad_edges.flatten(), return_counts = True)
        tip_vert = v[c == 2]
        other_vert = v[c == 1]
        print("tip vertex:", tip_vert)
        print("other vertex:", other_vert)


        # which two vertices should we consider connecting the tip vertex to?
        # a) find edges containing these other verts
        #print(edges[unique[counts == 1]])
        print( np.any(np.isin(edges[unique[counts == 1]], other_vert), axis = 1) )
        new_edges = edges[unique[counts == 1]][np.any(np.isin(edges[unique[counts == 1]], other_vert), axis = 1)]
        print("new_edges:", new_edges)

        points_to_consider, cc = np.unique(new_edges, return_counts = True)
        points_to_consider = points_to_consider[cc == 1]

        print("points_to_consider:", points_to_consider)

        best_point = points_to_consider[np.argmin(cdist(X[[tip_vert]],  X[points_to_consider]))]
        print("best_points:", best_point)

        test = new_edges[np.any(new_edges == best_point, axis = 1)].flatten() 
        print("test:", test)

        new_element = np.hstack([test, tip_vert])
        print("new_element:", new_element)

        input("[WAIT]")

        Tri = np.vstack([Tri, new_element])

        if False:
            import pyvista as pv
            plotter = pv.Plotter()

            plt_surf = pv.PolyData(X, np.hstack([ np.full(Tri.shape[0], 3)[:,None] , Tri ]))
            plotter.add_mesh(plt_surf, show_edges = True, opacity = 1.0) # add color to mesh here

            plt_points = pv.PolyData(X[dodgy_face])
            plotter.add_mesh(plt_points, color = "red", point_size = 10, render_points_as_spheres = True) # add color to mesh here

            plotter.show()

    return Tri

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


