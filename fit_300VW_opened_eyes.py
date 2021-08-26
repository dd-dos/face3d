from face3d.utils import draw_landmarks
import os
import multiprocessing as mp
from pathlib import Path
from face3d.face_model import FaceModel
import tqdm
import cv2
import scipy.io as sio
import glob
import numpy as np
import shutil
import utils

if __name__=='__main__':
    shutil.rmtree(f'300VW-3D_cropped_opened_eyes_3ddfa', ignore_errors=True)
    os.makedirs(f'300VW-3D_cropped_opened_eyes_3ddfa', exist_ok=True)
    for folder_path in glob.glob('300VW-3D_cropped_opened_eyes/*'):
        folder_img_name = folder_path.split('/')[-1]
        os.makedirs(
            os.path.join(f'300VW-3D_cropped_opened_eyes_3ddfa', folder_img_name),
            exist_ok=True
        )

    model = FaceModel()
    img_list = list(Path('300VW-3D_cropped_opened_eyes').glob('**/*.jpg'))
    bag = []
    print(f'Push item to bag: ')
    for img_path in tqdm.tqdm(img_list):
        pts_path = str(img_path).replace('jpg', 'mat')
        bag.append((str(img_path), pts_path))

    def task(item, debug):
        img_path, pts_path = item
        img_name = img_path.split('/')[-1].split('.')[0]
        folder_name = img_path.split('/')[-2]
        img = cv2.imread(img_path)
        pts = sio.loadmat(pts_path)['pt3d']

        output = model.generate_3ddfa_params_plus(img, pts, expand_ratio=1., preprocess=False, horizontal=[-50, -30, 0, 30, 50], vertical=[-70, -60, -50, 50])
        for idx in range(len(output)):
            ori_img = output[idx][0]
            ori_params = output[idx][1]

            img_out_path = os.path.join(f'300VW-3D_cropped_opened_eyes_3ddfa/{folder_name}', f'{img_name}_{idx}.jpg')
            params_out_path = os.path.join(f'300VW-3D_cropped_opened_eyes_3ddfa/{folder_name}', f'{img_name}_{idx}.mat')
            cv2.imwrite(img_out_path, ori_img)
            sio.savemat(params_out_path, ori_params)

            if debug:
                vertex = model.reconstruct_vertex(ori_img, ori_params['params'], de_normalize=False)[:,:2][model.bfm.kpt_ind]
                draw_landmarks(ori_img.copy(), vertex.copy(), f'debug/1_{folder_name}_{img_name}_{idx}.jpg')

        fliplr_img, fliplr_pts = utils.fliplr_face_landmarks(img, pts)
        # draw_landmarks(fliplr_img.copy(), fliplr_pts.copy(), f'intermediate.jpg')

        fliplr_output = model.generate_3ddfa_params_plus(fliplr_img, fliplr_pts, expand_ratio=1., preprocess=False, horizontal=[-50, -30, 0, 30, 50], vertical=[-70, -60, -50, 50])
        for idx in range(len(fliplr_output)):
            fliplr_img = fliplr_output[idx][0]
            fliplr_params = fliplr_output[idx][1]

            fliplr_img_out_path = os.path.join(f'300VW-3D_cropped_opened_eyes_3ddfa/{folder_name}', f'{img_name}_{idx}_fliplr.jpg')
            fliplr_params_out_path = os.path.join(f'300VW-3D_cropped_opened_eyes_3ddfa/{folder_name}', f'{img_name}_{idx}_fliplr.mat')
            cv2.imwrite(fliplr_img_out_path, fliplr_img)
            sio.savemat(fliplr_params_out_path, fliplr_params)

            if debug:
                vertex = model.reconstruct_vertex(fliplr_img, fliplr_params['params'], de_normalize=False)[:,:2][model.bfm.kpt_ind]
                draw_landmarks(fliplr_img.copy(), vertex.copy(), f'debug/2_{folder_name}_{img_name}_{idx}_fliplr.jpg')
    
    debug = True
    shutil.rmtree('debug', ignore_errors=True)
    os.makedirs('debug', exist_ok=True)
    for idx in tqdm.tqdm(range(len(bag)), total=len(bag)):
        if idx % 1000 == 0:
            debug = True
            task(bag[idx], debug)
            debug = False
        else:
            task(bag[idx], debug)