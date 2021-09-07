from multiprocessing.pool import IMapUnorderedIterator
import os
import multiprocessing as mp
from pathlib import Path

from numba.core.utils import format_time
from face3d.face_model import FaceModel
import tqdm
import cv2
import scipy.io as sio
import glob
import numpy as np
import utils
from utils import draw_pts
import shutil
import random

if __name__=='__main__':
    shutil.rmtree('300WLP_3ddfa', ignore_errors=True)
    os.makedirs('300WLP_3ddfa', exist_ok=True)
    os.makedirs('300WLP_3ddfa/300WLP_3ddfa-verified', exist_ok=True)

    model = FaceModel()
    img_list = list(Path('300WLP').glob('**/*.jpg'))
    bag = []
    print(f'Push item to bag: ')
    for img_path in tqdm.tqdm(img_list):
        pts_path = str(img_path).replace('jpg', 'mat')
        bag.append((str(img_path), pts_path))

    def task(item):
        img_path, pts_path = item
        name = img_path.split('/')[-1].split('.')[0]
        original_img = cv2.imread(img_path)
        original_pts = sio.loadmat(pts_path)['pt3d']

        img, info = model.generate_3ddfa_params(original_img, original_pts, False, expand_ratio=1.)
        img_out_path = os.path.join('300WLP_3ddfa/300WLP_3ddfa-verified', f'{name}.jpg')
        params_out_path = os.path.join('300WLP_3ddfa/300WLP_3ddfa-verified', f'{name}.mat')
        cv2.imwrite(img_out_path, img)
        sio.savemat(params_out_path, info)

        # fliplr_img, fliplr_pts = utils.fliplr_face_landmarks(original_img, original_pts, reverse=False)
        # expand_ratio = random.uniform(1., 1.4)
        # fliplr_img, fliplr_params = model.generate_3ddfa_params(fliplr_img, fliplr_pts, expand_ratio=expand_ratio)
        # fliplr_img_out_path = os.path.join('300WLP_3ddfa/300WLP_3ddfa-verified', f'{name}_fliplr.jpg')
        # fliplr_params_out_path = os.path.join('300WLP_3ddfa/300WLP_3ddfa-verified', f'{name}_fliplr.mat')
        # cv2.imwrite(fliplr_img_out_path, fliplr_img)
        # sio.savemat(fliplr_params_out_path, fliplr_params)

        # foo_pts = model.reconstruct_vertex(fliplr_img, fliplr_params['params'], de_normalize=False)[:,:2][model.bfm.kpt_ind]
        # for pts in foo_pts:
        #     pts = pts.astype(int)
        #     fliplr_img = cv2.circle(fliplr_img, pts,2,(0,255,0), -1, 8)
        # cv2.imwrite(f'test_flip_image.jpg', fliplr_img)

        # foo_pts = model.reconstruct_vertex(img, params['params'], de_normalize=False)[:,:2][model.bfm.kpt_ind]
        # for pts in foo_pts:
        #     pts = pts.astype(int)
        #     img = cv2.circle(img, pts,2,(0,255,0), -1, 8)
        # cv2.imwrite(f'test_images.jpg', img)
        
        # import ipdb; ipdb.set_trace(context=10)
        # import ipdb; ipdb.set_trace(context=10)
        # cv2.imshow('', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # with mp.Pool(2) as p:
    #     r = list(tqdm.tqdm(p.imap(task, bag), total=len(bag)))
    for item in tqdm.tqdm(bag):
        task(item)