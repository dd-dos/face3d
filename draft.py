import glob
import math
import pickle
import random
import shutil

import cv2
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy.io as sio
import tqdm
from PIL import Image

from face3d import mesh
from face3d.morphable_model import MorphabelModel
from face3d.utils import (crop_face_landmarks, resize_face_landmarks,
                          show_ndarray_img, show_pts, show_vertices)

bfm = MorphabelModel('/home/pdd/Desktop/Workspace/3DDFA-1/face3d/examples/Data/BFM/Out/BFM.mat')

# @numba.njit()
def get_colors(img, img_vertices):
    colors = np.zeros((len(img_vertices),3))

    for idx in range(len(img_vertices)):
        x, y = img_vertices[idx][:2]
        colors[idx] = img[y][x] / 255.

    return colors

def pt3d_to_3dmm():
    # img = cv2.imread('examples/Data/300WLP-std_134212_1_0.jpg')
    # pt = sio.loadmat('examples/Data/300WLP-std_134212_1_0.mat')['pt3d']
    # img, pt = crop_face_landmarks(img, pt, 1.5)
    # img, pt = resize_face_landmarks(img, pt)
    img = cv2.imread('examples/Data/image00050.jpg')
    pt = sio.loadmat('examples/Data/image00050.mat')['pt3d_68'][:2].T

    h,w,_ = img.shape
    pt.T[1] = h - 1 - pt.T[1]
    pt.T[0] -= w/2
    pt.T[1] -= h/2

    shp, exp, scale, angles, trans = bfm.reduced_fit(pt, n_sp=60, n_ep=29, max_iter=3)
    vertices = bfm.reduced_generated_vertices(shp, exp)

    transformed_vertices = bfm.transform(vertices, scale, angles, trans)

    image_vertices = mesh.transform.to_image(transformed_vertices, h, w)

    colors = get_colors(img, image_vertices.astype(np.uint16))

    angles = np.asarray(angles)
    rotate_list = [[-60., 20., 10]]
    angles += np.array(random.choice(rotate_list))
    
    # transformed_vertices = bfm.transform(vertices, scale, angles, trans)
    angles = mesh.transform.angle2matrix(angles)

    vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]
    scale = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1])) 
    trans = [0, 0, 0]
    transformed_vertices = mesh.transform.similarity_transform(vertices, scale, angles, trans)
    image_vertices = mesh.transform.to_image(transformed_vertices, h, w)

    image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)
    cv2_img = (image*255).astype(np.uint8)
    # cv2.imwrite(f'examples/results/{name}_rot.jpg', cv2_img)
    # shutil.copyfile(img_path, f'examples/results/{name}.jpg')
    show_ndarray_img(cv2_img)


if __name__=='__main__':
    import time

    # while True:
    #     t0 = time.time()
    #     pt3d_to_3dmm()
    #     print(time.time()-t0)
    #     break
    from face3d import face_model
    model = face_model.FaceModel(
        bfm_path='/home/pdd/Desktop/Workspace/3DDFA-1/face3d/examples/Data/BFM/Out/BFM.mat'
    )
    img = cv2.imread('examples/Data/300WLP-std_134212_1_0.jpg')
    pt = sio.loadmat('examples/Data/300WLP-std_134212_1_0.mat')['pt3d']
    # img = cv2.imread('examples/Data/image00050.jpg')
    # pt = sio.loadmat('examples/Data/image00050.mat')['pt3d_68'][:2].T
    r_img, params = model.generate_rotated_3d_img(img, pt)
    r_pt = model.reconstruct_vertex(r_img, params)
    show_pts(r_img, r_pt[model.bfm.kpt_ind])
