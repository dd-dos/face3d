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
                          show_ndarray_img, show_pts, show_vertices)

model = face_model.FaceModel()

def generated_rotated_sample(height, 
                            width, 
                            params=None, 
                            de_norm=False, 
                            adjusted_angles=[0, 0, 0],
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
    img = cv2.imread('AFLW2000_3ddfa/image00002.jpg')
    height, width = img.shape[:2]

    background = cv2.imread('hair.jpeg')
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    background = cv2.resize(background, (width, height))

    params = sio.loadmat('AFLW2000_3ddfa/image00002.mat')['params'].reshape(-1,)
    new_img, new_pts = generated_rotated_sample(height, width)
    new_img = cv2.cvtColor((new_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite('generated.jpg', new_img)
    # sio.savemat('generated.mat', {'pt3d': new_pts[:,:2][model.bfm.kpt_ind]})
    show_ndarray_img(new_img)
    # show_pts(new_img, new_pts[:,:2][model.bfm.kpt_ind])