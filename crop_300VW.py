import torchfile 
from pathlib import Path
import cv2
import tqdm
import numpy as np
import os
import multiprocessing as mp
import utils
import scipy.io as sio

def task(img_path):
    img_path = str(img_path)
    pts_3D_path = img_path.replace('jpg', 't7')

    token = img_path.split('/')
    token[0] = '300VW_Dataset_2015_12_14'
    token[-1] = '00'+token[-1].replace('jpg', 'pts')
    token.insert(2, 'annot')
    pts_2D_path = '/'.join(token)

    try:
        img = cv2.imread(img_path)
        pts_3D = torchfile.load(pts_3D_path)
        pts_2D = utils.read_pts(pts_2D_path)
    except:
        return

    img, pts_2D, pts_3D = utils.crop_face_landmarks(img, pts_2D, pts_3D, expand_ratio=1.1)
    img, pts_2D, pts_3D = utils.resize_face_landmarks(img, pts_2D, pts_3D, shape=(128,128))
    # pts_3D = utils.replace_eyes(pts_2D, pts_3D)
    
    token = img_path.split('/')
    folder_id = token[-2]
    file_id = token[-1]

    os.makedirs(f'300VW-3D_cropped/{folder_id}', exist_ok=True)
    out_path = f'300VW-3D_cropped/{folder_id}/{file_id}'
    cv2.imwrite(out_path, img)
    sio.savemat(f'300VW-3D_cropped/{folder_id}/{file_id}'.replace('jpg', 'mat'), {'pt3d': pts_3D, 'pt2D': pts_2D})

if __name__=='__main__':
    img_list = list(Path('300VW-3D_cropped').glob('**/*.jpg'))
    # with mp.Pool(2) as p:
    #     r = list(tqdm.tqdm(p.imap(task, img_list), total=len(img_list)))

    # for item in tqdm.tqdm(img_list):
    #     task(item)
    print(len(img_list))