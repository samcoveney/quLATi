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

from qulati.meshutils import extendMesh, subDivide


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


# main function
def eigensolver(X, Tri, holes, num = 2**8, layers = 10, calc_gradV = True, use_average_edge = False):
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
    X, Tri, edges, centroids = extendMesh(X, Tri, layers, holes, use_average_edge = use_average_edge)
    #np.save("/tmp/X.npy", X)
    #np.save("/tmp/Tri.npy", Tri)
    #input("waiting after saving extended mesh to /tmp")
    extended_num_verts = X.shape[0] # save number of vertices of extended mesh

    # 2. subdivide the mesh
    # ---------------------
    newX, newTri, vertsInFace = subDivide(X, Tri, edges, centroids)
    #newX, newTri = X, Tri # for testing without subdivide

    # 3. solve eigenproblem on the extended and subdivided mesh
    # ---------------------------------------------------------
    Q, V = LaplacianEigenpairs(newX, newTri, num = num)
    

    if calc_gradV:
        # 4. get eigenfunction gradients at mesh centroids
        # ------------------------------------------------
        gradV = gradientEigenvectors(newX, newTri, vertsInFace, V)


        # keep V only at original mesh vertices and centroids
        V = np.vstack( [ V[0:orig_num_verts] , V[extended_num_verts:extended_num_verts+orig_num_faces] ] )

        return Q, V[0:orig_num_verts+orig_num_faces], gradV[0:orig_num_faces, :, :], newX[extended_num_verts:extended_num_verts+orig_num_faces]

    else:

        return Q, V[0:orig_num_verts+orig_num_faces], None, newX[extended_num_verts:extended_num_verts+orig_num_faces]


