from PIL.Image import new
from matplotlib.pyplot import draw, show
from face3d.mesh import render, transform
import cv2
import numba
import numpy as np
import scipy.io as sio

from face3d import face_model, mesh
from face3d.utils import (crop_face_landmarks, isgray, resize_face_landmarks,
                          show_ndarray_img, show_pts, show_vertices, draw_landmarks)
import utils

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
    colors = face_model._get_colors(img, vertices.astype(np.uint16))

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

    # tp = model.bfm.get_tex_para('random')
    rendering = mesh.render.render_colors(
                    image_vertices, 
                    model.bfm.triangles, 
                    colors, 
                    height, 
                    width,
                    BG=background
                )

    rendering = np.minimum((np.maximum(rendering, 0)), 1)

    return rendering, image_vertices

if __name__=='__main__':
    # img, vertex = model._preprocess_face_landmarks(img, vertex)
    # rimg, _ = model.augment_rotate(img, vertex, [-70,0,0], base_size=180*0.7, de_normalize=False)
    img = cv2.imread('300WLP/300WLP-verified/300WLP-std_image_train_0871_3.jpg')
    vertex_3d = sio.loadmat('300WLP/300WLP-verified/300WLP-std_image_train_0871_3.mat')['pt3d']

    # vertex_2d = sio.loadmat('samples/001_original/0575.mat')['pt2d']
    # vertex_3d = utils.replace_eyes(vertex_2d, vertex_3d)
    # show_pts(img.copy(), vertex_3d.copy(), 'BGR')

    # output = model.generate_3ddfa_params_plus(
    #     img, vertex_3d, preprocess=False, horizontal=[0], vertical=[0])
    
    # for out in output:
    #     rimg = out[0]
    #     params = out[1]['params']
    #     vertex = model.reconstruct_vertex(rimg, params, False)[model.bfm.kpt_ind]
    #     show_pts(rimg, vertex, 'BGR')

    draw_landmarks(img, vertex_3d)