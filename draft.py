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
from face3d import utils
import face3d
from face3d.morphable_model import MorphabelModel
from face3d.utils import (crop_face_landmarks, isgray, resize_face_landmarks,
                          show_ndarray_img, show_pts, show_vertices)
import torch

if __name__=='__main__':
    import time

    # while True:
    #     t0 = time.time()
    #     pt3d_to_3dmm()
    #     print(time.time()-t0)
    #     break
    from face3d import face_model
    model = face_model.FaceModel()
    # img = cv2.imread('examples/Data/300WLP-std_134212_1_0.jpg')
    # pt = sio.loadmat('examples/Data/300WLP-std_134212_1_0.mat')['pt3d']
    # img = cv2.imread('AFLW2000/image01649.jpg')
    # pt = sio.loadmat('AFLW2000/image01649.mat')['pt3d_68'][:2].T
    # img, params = model.generate_rotated_3d_img(img, pt)
    img = cv2.imread('300VW-3D_cropped/505/0037.jpg')
    pt = sio.loadmat('300VW-3D_cropped/505/0037.mat')['pt2D']
    for i in range(pt.shape[0]):
        _pts = pt[i].astype(int)
        _img = cv2.circle(img, (_pts[0], _pts[1]),2,(0,255,0), -1, 5)
    
    cv2.imwrite(f'test_images.jpg', _img)


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

    