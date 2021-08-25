from PIL.Image import new
from matplotlib.pyplot import show
from face3d.mesh import render, transform
import cv2
import numba
import numpy as np
import scipy.io as sio

from face3d import face_model, mesh, utils
from face3d.morphable_model import MorphabelModel
from face3d.utils import (crop_face_landmarks, isgray, resize_face_landmarks,
                          show_ndarray_img, show_pts, show_vertices, draw_landmarks)

model = face_model.FaceModel()

def generated_rotated_sample(height, 
                            width, 
                            params=None, 
                            de_norm=False, 
                            adjusted_angles=[-40, 0, 0],
                            background=None,
                            ):
    if params is None:
        params = model.bfm.generate_params()

    shp, exp, scale, angles, trans = \
        model._parse_params(params, de_normalize=de_norm)

    vertices = model.bfm.reduced_generated_vertices(shp, exp)
    vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]

    new_angles = np.array(adjusted_angles)
    trans = [0,-10,0]
    scale = 6e-4

    transformed_vertices = model.bfm.transform(
        vertices, scale, new_angles, trans
    ) 

    image_vertices = mesh.transform.to_image(
        transformed_vertices, height, width
    )

    tp = model.bfm.get_tex_para('random')
    rendering = mesh.render.render_colors(
                    image_vertices, 
                    model.bfm.triangles, 
                    model.bfm.generate_colors(tp), 
                    height, 
                    width,
                    BG=background
                )

    rendering = np.minimum((np.maximum(rendering, 0)), 1)

    return rendering, image_vertices

if __name__=='__main__':
    img = cv2.imread('300VW-3D_cropped/001/0001.jpg')
    vertex = sio.loadmat('300VW-3D_cropped/001/0001.mat')['pt3d']
    # vertex = model.reconstruct_vertex(img, params, de_normalize=False)[:,:2][model.bfm.kpt_ind]

    import time
    t0 = time.time()
    img, vertex = model._preprocess_face_landmarks(img, vertex)
    rimg, _ = model.augment_rotate(img, vertex, [-70,0,0], base_size=180*0.7, de_normalize=False)
    print(time.time() - t0)
    show_ndarray_img(rimg)