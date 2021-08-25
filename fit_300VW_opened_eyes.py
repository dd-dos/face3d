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

    def task(item):
        img_path, pts_path = item
        img_name = img_path.split('/')[-1].split('.')[0]
        folder_name = img_path.split('/')[-2]
        img = cv2.imread(img_path)
        pts = sio.loadmat(pts_path)['pt3d']

        fliplr_img, fliplr_pts = utils.fliplr_face_landmarks(img, pts)

        output = model.generate_3ddfa_params_plus(img, pts, expand_ratio=1., preprocess=False, horizontal=[-50, -25, 0, 25, 50], vertical=[-70, -35, 35, 70])
        for idx in range(len(output)):
            img = output[idx][0]
            params = output[idx][1]

            img_out_path = os.path.join(f'300VW-3D_cropped_opened_eyes_3ddfa/{folder_name}', f'{img_name}_{idx}.jpg')
            params_out_path = os.path.join(f'300VW-3D_cropped_opened_eyes_3ddfa/{folder_name}', f'{img_name}_{idx}.mat')
            cv2.imwrite(img_out_path, img)
            sio.savemat(params_out_path, params)

        fliplr_output = model.generate_3ddfa_params_plus(fliplr_img, fliplr_pts, expand_ratio=1., preprocess=False, horizontal=[-50, -25, 0, 25, 50], vertical=[-70, -35, 35, 70])
        for idx in range(len(fliplr_output)):
            fliplr_img = fliplr_output[idx][0]
            fliplr_params = fliplr_output[idx][1]

            fliplr_img_out_path = os.path.join(f'300VW-3D_cropped_opened_eyes_3ddfa/{folder_name}', f'{img_name}_{idx}_fliplr.jpg')
            fliplr_params_out_path = os.path.join(f'300VW-3D_cropped_opened_eyes_3ddfa/{folder_name}', f'{img_name}_{idx}_fliplr.mat')
            cv2.imwrite(fliplr_img_out_path, fliplr_img)
            sio.savemat(fliplr_params_out_path, fliplr_params)

    for item in tqdm.tqdm(bag):
        task(item)