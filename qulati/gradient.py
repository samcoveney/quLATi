"""
   gradient.py

   Calculate gradients of scalar fields defined on vertices of a triangle mesh.

   Created: 11-Feb-2020
   Author:  Sam Coveney
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import trimesh


# CV stats function
def gradient(X, Tri, scalar, magnitude_stats = False):
    '''Calculate gradient of scalar field on every mesh face for all scaler fields (at vertices) provided.

       For N-vertex F-face mesh, scalar can be (N x m) array, so calculation is performed for m different scaler fields to get (F x n) array result.
    '''

    # trimesh object
    mesh = trimesh.Trimesh(vertices = X, faces = Tri, process = False)
    areas = mesh.area_faces
    normals = mesh.face_normals 


    # calculate "CVfaceSamples" - CV vector at element/face center for each scalar sample

    if scalar.ndim == 1: scalar = scalar.reshape([-1,1])  # if 1D array, reshape into 2D array

    print(scalar.shape[1], "scalar samples;", "looping over", mesh.faces.shape[0], "mesh faces... ")

    CVfaceSamples = np.zeros([mesh.faces.shape[0], scalar.shape[1], 3])


    # loop over mesh faces
    for ff in range(mesh.faces.shape[0]):

        # 1. get face vertices
        # --------------------
        face = mesh.faces[ff]

        # 2. get vertex coordinates
        # -------------------------
        vertices = mesh.vertices[(face[2], face[0], face[1]), :]  # store vertex coords as [vc, va, vb]

        # 3. normalize vectors, then divide by 2*area
        # -------------------------------------------
        B = np.diff(np.vstack([vertices[-1],vertices]), axis = 0)  # gives us [ [c-b, a-c, b-a] ]
        D = B / (2*areas[ff])  # do NOT turn edge vectors B into unit vectors

        # 4. cross product with face normals
        # -----------------------------------
        E = np.cross(normals[ff], D)  # if minus sign included, arrows point wrong way, so ignore minus

        # 5. calculate gradients of scalar
        # -----------------------------
        g = scalar[face,:]
        v = np.einsum('ki,kj->ij', g, E)
        CVfaceSamples[ff] = v


    return np.squeeze(CVfaceSamples)


