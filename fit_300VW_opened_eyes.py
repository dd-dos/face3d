import os
import multiprocessing as mp
from pathlib import Path

from numpy.random import rand
from face3d.face_model import FaceModel
import tqdm
import cv2
import scipy.io as sio
import glob
import numpy as np
import shutil
import utils
from face3d.utils import draw_landmarks
import random

if __name__=='__main__':
    shutil.rmtree(f'300VW-3D_opened_eyes_3ddfa', ignore_errors=True)
    os.makedirs(f'300VW-3D_opened_eyes_3ddfa', exist_ok=True)
    for folder_path in glob.glob('GANnotation/300VW-3D_opened_eyes/*'):
        folder_img_name = folder_path.split('/')[-1]
        os.makedirs(
            os.path.join(f'300VW-3D_opened_eyes_3ddfa', folder_img_name),
            exist_ok=True
        )

    model = FaceModel()
    img_list = list(Path('GANnotation/300VW-3D_opened_eyes').glob('**/*.jpg'))
    img_list = [img_list[idx] for idx in range(len(img_list)) if idx%3==0]

    bag = []
    print(f'Push item to bag: ')
    for img_path in tqdm.tqdm(img_list):
        pts_path = str(img_path).replace('jpg', 'mat')
        bag.append((str(img_path), pts_path))

    def task(item, debug):
        img_path, pts_path = item
        img_name = img_path.split('/')[-1].split('.')[0]
        folder_name = img_path.split('/')[-2]
        original_img = cv2.imread(img_path)
        original_pts = sio.loadmat(pts_path)['pt3d']

        expand_ratio = 1.
        yaw = np.random.choice([random.uniform(-40, -20), random.uniform(-20, 20), random.uniform(20, 40)], p=[0.3, 0.4, 0.3])
        pitch = np.random.choice([random.uniform(-55, -50), random.uniform(-50, -40), random.uniform(-40, 30), random.uniform(30, 40)], p=[0.15, 0.35, 0.15, 0.35])
        roll = 0.

        output = model.generate_3ddfa_params_plus(original_img, original_pts, expand_ratio=expand_ratio, preprocess=False, yaw=yaw, pitch=pitch, ignore_high_pitch=False)
        for idx in range(len(output)):
            ori_img = output[idx][0]
            ori_params = output[idx][1]

            img_out_path = os.path.join(f'300VW-3D_opened_eyes_3ddfa/{folder_name}', f'{img_name}_{idx}.jpg')
            params_out_path = os.path.join(f'300VW-3D_opened_eyes_3ddfa/{folder_name}', f'{img_name}_{idx}.mat')
            cv2.imwrite(img_out_path, ori_img)
            sio.savemat(params_out_path, ori_params)

            if debug:
                vertex = model.reconstruct_vertex(ori_img, ori_params['params'], de_normalize=False)[:,:2][model.bfm.kpt_ind]
                draw_landmarks(ori_img.copy(), vertex.copy(), f'debug/1_{folder_name}_{img_name}_{idx}.jpg')

        # fliplr_img, fliplr_pts = utils.fliplr_face_landmarks(original_img, original_pts, reverse=False)
        # # draw_landmarks(fliplr_img.copy(), fliplr_pts.copy(), f'intermediate.jpg')

        # expand_ratio = random.uniform(1,1.4)
        # yaw = np.random.choice([random.uniform(-50, -30), random.uniform(-30,30), random.uniform(30,50)])
        # pitch = np.random.choice([random.uniform(-70, -60), random.uniform(-60, 50)])

        # fliplr_output = model.generate_3ddfa_params_plus(fliplr_img, fliplr_pts, expand_ratio=expand_ratio, preprocess=True, yaw=yaw, pitch=pitch)
        # for idx in range(len(fliplr_output)):
        #     fliplr_img = fliplr_output[idx][0]
        #     fliplr_params = fliplr_output[idx][1]

        #     fliplr_img_out_path = os.path.join(f'300VW-3D_opened_eyes_3ddfa/{folder_name}', f'{img_name}_{idx}_fliplr.jpg')
        #     fliplr_params_out_path = os.path.join(f'300VW-3D_opened_eyes_3ddfa/{folder_name}', f'{img_name}_{idx}_fliplr.mat')
        #     cv2.imwrite(fliplr_img_out_path, fliplr_img)
        #     sio.savemat(fliplr_params_out_path, fliplr_params)

        #     if debug:
        #         vertex = model.reconstruct_vertex(fliplr_img, fliplr_params['params'], de_normalize=False)[:,:2][model.bfm.kpt_ind]
        #         draw_landmarks(fliplr_img.copy(), vertex.copy(), f'debug/2_{folder_name}_{img_name}_{idx}_fliplr.jpg')

    # custom_item = ('300VW-3D_cropped_opened_eyes/203/0818.jpg', '300VW-3D_cropped_opened_eyes/203/0818.mat')
    # task(custom_item, True)
    # import ipdb; ipdb.set_trace(context=10)
    debug = True
    shutil.rmtree('debug', ignore_errors=True)
    os.makedirs('debug', exist_ok=True)
    for idx in tqdm.tqdm(range(len(bag)), total=len(bag)):
        if idx % 100 == 0:
            debug = True
            task(bag[idx], debug)
            debug = False
        else:
            task(bag[idx], debug)