import os
import glob
from pathlib import Path

import utils
import shutil
import tqdm
import scipy.io as sio
import multiprocessing as mp
import cv2
import numpy as np

def task(item):
    item = str(item)
    pts_2d = sio.loadmat(item)['pt2D']
    eyes_2d = utils.get_eyes(pts_2d)


    folder_id = item.split('/')[-2]
    file_id = item.split('/')[-1]

    if utils.check_eye_status(eyes_2d['left']) == 'closed' and utils.check_eye_status(eyes_2d['right']) == 'closed':
        shutil.copyfile(item.replace('mat', 'jpg'), f'300VW-3D_cropped_closed_eyes/{folder_id}/{file_id}.jpg')
        shutil.copyfile(item, f'300VW-3D_cropped_closed_eyes/{folder_id}/{file_id}.mat')
        pts_3d = sio.loadmat(item)['pt3d']
        pts_3d = utils.replace_eyes(pts_2d, pts_3d)

if __name__=='__main__':
    shutil.rmtree('300VW-3D_cropped_closed_eyes', ignore_errors=True)
    shutil.rmtree('test_images', ignore_errors=True)

    os.makedirs('300VW-3D_cropped_closed_eyes', exist_ok=True)
    os.makedirs('test_images', exist_ok=True)

    bag = list(Path('../300VW-3D_cropped').glob('**/*.mat'))

    for folder_path in glob.glob('../300VW-3D_cropped/*'):
        folder_name = folder_path.split('/')[-1]
        os.makedirs(
            os.path.join('300VW-3D_cropped_closed_eyes', folder_name),
            exist_ok=True
        )
        os.makedirs(
            os.path.join('test_images', folder_name),
            exist_ok=True
        )
    with mp.Pool(mp.cpu_count()) as p:
        r = list(tqdm.tqdm(p.imap(task, bag), total=len(bag)))