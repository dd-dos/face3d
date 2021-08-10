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
    # from face3d import face_model
    # model = face_model.FaceModel(
    #     bfm_path='/home/pdd/Desktop/Workspace/3DDFA-1/face3d/examples/Data/BFM/Out/BFM.mat'
    # )
    # img = cv2.imread('examples/Data/300WLP-std_134212_1_0.jpg')
    # pt = sio.loadmat('examples/Data/300WLP-std_134212_1_0.mat')['pt3d']
    # img = cv2.imread('AFLW2000/image01649.jpg')
    # pt = sio.loadmat('AFLW2000/image01649.mat')['pt3d_68'][:2].T
    # img, params = model.generate_rotated_3d_img(img, pt)
    # img = cv2.imread('300WLP_processed/300WLP_processed-verified/300WLP_processed-std_4970182488_1_6.jpg')
    # params = np.load('300WLP_processed/300WLP_processed-verified/300WLP_processed-std_4970182488_1_6.npy')
    # pt = model.reconstruct_vertex(img, params)
    # show_pts(img, pt[model.bfm.kpt_ind])

    ################################################################################

    from face3d.tddfa.denseface import FaceAlignment
    from face3d.utils import crop_face_landmarks

    model = FaceAlignment('face3d/tddfa/weights/2021-07-22/last.pth.tar', device='cpu')

    img = cv2.imread('AFLW2000/image00050.jpg')
    pt = sio.loadmat('AFLW2000/image00050.mat')['pt3d_68'][:2].T
    img, pt = crop_face_landmarks(img, pt)
    h, w, _ = img.shape
    n_img = model.draw_landmarks(img, [torch.tensor([0,0,w,h])])
    utils.show_ndarray_img(n_img)

    