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
    pt2d = sio.loadmat(item)['pt2d']
    eyes_2d = utils.get_eyes(pt2d)


    folder_id = item.split('/')[-2]
    file_id = item.split('/')[-1].split('.')[0]

    if utils.check_eye_status(eyes_2d['left']) == 'closed' and utils.check_eye_status(eyes_2d['right']) == 'closed':
        shutil.copyfile(item.replace('mat', 'jpg'), f'300VW-3D_closed_eyes/{folder_id}/{file_id}.jpg')
        pt3d = sio.loadmat(item)['pt3d']
        pt3d = utils.replace_eyes(pt2d, pt3d)
        sio.savemat(f'300VW-3D_closed_eyes/{folder_id}/{file_id}.mat', {'pt3d': pt3d})
    elif utils.check_eye_status(eyes_2d['left']) == 'opened' and utils.check_eye_status(eyes_2d['right']) == 'opened':
        shutil.copyfile(item.replace('mat', 'jpg'), f'300VW-3D_opened_eyes/{folder_id}/{file_id}.jpg')
        pt3d = sio.loadmat(item)['pt3d']
        pt3d = utils.replace_eyes(pt2d, pt3d)
        sio.savemat(f'300VW-3D_opened_eyes/{folder_id}/{file_id}.mat', {'pt3d': pt3d})
    elif utils.check_eye_status(eyes_2d['left']) == 'semi' and utils.check_eye_status(eyes_2d['right']) == 'semi':
        shutil.copyfile(item.replace('mat', 'jpg'), f'300VW-3D_semi_eyes/{folder_id}/{file_id}.jpg')
        pt3d = sio.loadmat(item)['pt3d']
        pt3d = utils.replace_eyes(pt2d, pt3d) 
        sio.savemat(f'300VW-3D_semi_eyes/{folder_id}/{file_id}.mat', {'pt3d': pt3d})

if __name__=='__main__':
    shutil.rmtree('300VW-3D_closed_eyes', ignore_errors=True)
    shutil.rmtree('300VW-3D_opened_eyes', ignore_errors=True)
    shutil.rmtree('300VW-3D_semi_eyes', ignore_errors=True)
    shutil.rmtree('test_images', ignore_errors=True)

    os.makedirs('300VW-3D_closed_eyes', exist_ok=True)
    os.makedirs('300VW-3D_opened_eyes', exist_ok=True)
    os.makedirs('300VW-3D_semi_eyes', exist_ok=True)
    os.makedirs('test_images', exist_ok=True)

    bag = list(Path('../300VW-3D_and_2D').glob('**/*.mat'))

    for folder_path in glob.glob('../300VW-3D_and_2D/*'):
        folder_name = folder_path.split('/')[-1]
        os.makedirs(
            os.path.join('300VW-3D_closed_eyes', folder_name),
            exist_ok=True
        )
        os.makedirs(
            os.path.join('300VW-3D_opened_eyes', folder_name),
            exist_ok=True
        )
        os.makedirs(
            os.path.join('300VW-3D_semi_eyes', folder_name),
            exist_ok=True
        )
        os.makedirs(
            os.path.join('test_images', folder_name),
            exist_ok=True
        )
    with mp.Pool(mp.cpu_count()) as p:
        r = list(tqdm.tqdm(p.imap(task, bag), total=len(bag)))

