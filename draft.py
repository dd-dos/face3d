import cv2
import numba
import numpy as np
import scipy.io as sio
import torch
import tqdm
from PIL import Image

import face3d
from face3d import mesh, utils
from face3d.morphable_model import MorphabelModel
from face3d.utils import (crop_face_landmarks, isgray, resize_face_landmarks,
                          show_ndarray_img, show_pts, show_vertices)

@numba.njit()
def shift(img, params, shift_value=(-20,30)):
    img_height, img_width = img.shape[:2]
    canvas = np.zeros((img_height*2,img_width*2,3))
    crop_size = int(img_height/2)
    canvas[crop_size:img_height+crop_size, crop_size:img_width+crop_size, :] = img
    
    new_x1 = crop_size+shift_value[0]
    new_y1 = crop_size+shift_value[1]
    new_x2 = crop_size+img_width+shift_value[0]
    new_y2 = crop_size+img_height+shift_value[1]
    
    new_img = canvas[new_y1:new_y2, new_x1:new_x2]
    params[3] -= shift_value[0]
    params[7] += shift_value[1]

    return new_img, params

if __name__=='__main__':
    import time

    # while True:
    #     t0 = time.time()
    #     pt3d_to_3dmm()
    #     print(time.time()-t0)
    #     break
    from face3d import face_model
    model = face_model.FaceModel()
    img = cv2.imread('examples/Data/300WLP-std_134212_1_0.jpg')
    pt = sio.loadmat('examples/Data/300WLP-std_134212_1_0.mat')['pt3d']
    # pt[37] = pt[41]
    # pt[38] = pt[40]
    # pt[43] = pt[47]
    # pt[44] = pt[46]
    
    img, params = model.generate_3ddfa_params(img, pt, expand_ratio=0.95)
    params = params['params'].reshape(101,)
    # img, params = shift(img, params)
    new_pt = model.reconstruct_vertex(img, params)
    # img = cv2.imread('AFLW2000/image01649.jpg')
    # pt = sio.loadmat('AFLW2000/image01649.mat')['pt3d_68'][:2].T
    # img, params = model.generate_rotated_3d_img(img, pt)
    # img = cv2.imread('300VW-3D_cropped_3ddfa/519/1692.jpg')
    # params = sio.loadmat('300VW-3D_cropped_3ddfa/519/1692.mat')['params']
    # params[11] = 100000
    # pt = model.reconstruct_vertex(img, params)
    # show_vertices(new_pt[model.bfm.kpt_ind], '2D')
    show_pts(img, new_pt[model.bfm.kpt_ind])

    ################################################################################

    # from face3d.tddfa.denseface import FaceAlignment
    # from face3d.utils import crop_face_landmarks

    # model = FaceAlignment('face3d/tddfa/weights/2021-07-22/last.pth.tar', device='cpu')

    # img = cv2.imread('AFLW2000/image00050.jpg')
    # pt = sio.loadmat('AFLW2000/image00050.mat')['pt3d_68'][:2].T
    # img, pt = crop_face_landmarks(img, pt)
    # h, w, _ = img.shape
    # n_img = model.draw_landmarks(img, [torch.tensor([0,0,w,h])])
    # utils.show_ndarray_img(n_img)

    