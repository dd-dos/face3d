import shutil
import torchfile 
from pathlib import Path
import cv2
import tqdm
import numpy as np
import os
import multiprocessing as mp
import utils
import scipy.io as sio
import glob

def task(img_path):
    img_path = str(img_path)
    pts_3d_path = img_path.replace('jpg', 't7')

    token = img_path.split('/')
    token[0] = '300VW_Dataset_2015_12_14'
    token[-1] = '00'+token[-1].replace('jpg', 'pts')
    token.insert(2, 'annot')
    pts_2d_path = '/'.join(token)

    try:
        img = cv2.imread(img_path)
        pts_3d = torchfile.load(pts_3d_path)
        pts_2d = utils.read_pts(pts_2d_path)
    except Exception as e:
        print(e)
        return

    # img, pts_2d, pts_3d = utils.crop_multi_face_landmarks(img, pts_2d, pts_3d, expand_ratio=1.)
    # img, pts_2d, pts_3d = utils.resize_face_landmarks(img, pts_2d, pts_3d, shape=(128,128))
    
    token = img_path.split('/')
    folder_id = token[-2]
    file_id = token[-1]

    out_path = f'300VW-3D_and_2D/{folder_id}/{file_id}'
    cv2.imwrite(out_path, img)
    sio.savemat(f'300VW-3D_and_2D/{folder_id}/{file_id}'.replace('jpg', 'mat'), {'pt3d': pts_3d, 'pt2d': pts_2d})

if __name__=='__main__':
    shutil.rmtree('300VW-3D_and_2D', ignore_errors=True)
    os.makedirs('300VW-3D_and_2D', exist_ok=True)
    for folder_path in glob.glob('300VW-3D/*'):
        folder_img_name = folder_path.split('/')[-1]
        os.makedirs(
            os.path.join(f'300VW-3D_and_2D', folder_img_name),
            exist_ok=True
        )

    img_list = list(Path('300VW-3D').glob('**/*.jpg'))
    with mp.Pool(mp.cpu_count()) as p:
        r = list(tqdm.tqdm(p.imap(task, img_list), total=len(img_list)))
    # for item in tqdm.tqdm(img_list):
    #     task(item)