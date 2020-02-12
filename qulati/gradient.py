"""
   gradient.py

   Calculate gradients of scalar fields defined on vertices of a triangle mesh.

   Created: 11-Feb-2020
   Author:  Sam Coveney
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import pymesh


# CV stats function
def gradient(X, Tri, scalar, at_vertices = True, magnitude_stats = True):
    '''Calculate gradient of scalar field on every mesh *vertex* (from average of of gradients at mesh elements) for all scaler fields provided.

       NOTE:
       * for N-vertex mesh, scalar can be (N x m) array, so calculation is performed for m different scaler fields.
    '''


    #{{{ create mesh attributes

    # PyMesh object
    mesh = pymesh.form_mesh(X, Tri)

    # area
    mesh.add_attribute("face_area")
    areas = mesh.get_attribute("face_area")

    # normals
    mesh.add_attribute("face_normal")
    normals = mesh.get_face_attribute("face_normal")  # N.B. have not used get_attribute()
    
    mesh.add_attribute("vertex_normal")
    vertexNormals = mesh.get_attribute("vertex_normal").reshape(-1,3)

    # face IDs
    mesh.add_attribute("face_index")
    IDs = mesh.get_attribute("face_index").astype(int)

    # face centroid
    mesh.add_attribute("face_centroid")
    centroids = mesh.get_attribute("face_centroid").reshape(-1,3)

    #}}}


    #{{{ calculate "CVfaceSamples" - CV vector at element/face center for each scalar sample

    if scalar.ndim == 1: scalar = scalar.reshape([-1,1])  # if 1D array, reshape into 2D array

    print(scalar.shape[1], "scalar samples;", "looping over", IDs.shape[0], "mesh faces... ")

    CVfaceSamples = np.zeros([IDs.shape[0], scalar.shape[1], 3]) # NOTE: store CV vector for all samples at every face

    for faceID in IDs:

        # 1. get face vertices
        # --------------------
        vertexID = mesh.faces[faceID]

        # 2. get vertex coordinates
        # -------------------------
        vertices = mesh.vertices[(vertexID[2], vertexID[0], vertexID[1]), :]  # store vertex coords as [vc, va, vb]

        # 3. normalize vectors, then divide by 2*area
        # -------------------------------------------
        B = np.diff(np.vstack([vertices[-1],vertices]), axis = 0)  # gives us [ [c-b, a-c, b-a] ]
        D = B / (2*areas[faceID])  # do NOT turn edge vectors B into unit vectors

        # 4. cross product with face normals
        # -----------------------------------
        E = np.cross(normals[faceID], D)  # if minus sign included, arrows point wrong way, so ignore minus

        # 5. calculate gradients of scalar
        # -----------------------------
        g = scalar[vertexID,:]
        v = np.einsum('ki,kj->ij', g, E)
        CVfaceSamples[faceID] = v

    #}}}

    
    #{{{ calculate "CVvertexSamples" - CV vector at vertex for each scalar sample
   
    CVvertexSamples = np.zeros([mesh.vertices.shape[0], scalar.shape[1], 3]) # NOTE: store CV vector for all samples at every vertex

    for s in range(scalar.shape[1]):

        CV = CVfaceSamples[:,s,:]

        vertexCVx = pymesh.convert_to_vertex_attribute(mesh, CV[:,0]).reshape([-1,1])
        vertexCVy = pymesh.convert_to_vertex_attribute(mesh, CV[:,1]).reshape([-1,1])
        vertexCVz = pymesh.convert_to_vertex_attribute(mesh, CV[:,2]).reshape([-1,1])

        CVvertexSamples[:,s,:] = np.hstack([vertexCVx, vertexCVy, vertexCVz])

    #}}}


    #{{{ decide upon and return results

    grad_results = CVvertexSamples if at_vertices else CVfaceSamples

    if magnitude_stats == False:

        return np.squeeze(grad_results)

    else: 

        mag = np.linalg.norm(grad_results, axis = 2)

        mean = np.mean(mag, axis = 1)
        stdev = np.std(mag, axis = 1)

        if scalar.shape[1] > 1:

            qs = np.percentile(mag, [9, 25, 50, 75, 91], axis = 1).T

            llq, lq, median, hq, hhq = qs[0], qs[1], qs[2], qs[3], qs[4]
            
            print(stdev[:,None].shape)
            print(qs.shape)

            stats_results = np.hstack([ mean[:,None], stdev[:,None], qs ])

        else:
            
            stats_results = np.hstack([ mag, np.full([mag.shape[0], 6], np.nan) ])

        return stats_results

    #}}}


