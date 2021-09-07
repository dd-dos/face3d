import os
from pathlib import Path

from numpy.core.numeric import full

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
from face3d.utils import resize_face_landmarks, get_landmarks_wrapbox
from face3d.utils import draw_landmarks

fm = FaceModel()

if __name__=='__main__':
    shutil.rmtree('squeezed_face_3ddfa', ignore_errors=True)
    os.makedirs('squeezed_face_3ddfa', exist_ok=True)
    os.makedirs('squeezed_face_3ddfa/300WLP-verified', exist_ok=True)
    os.makedirs('squeezed_face_3ddfa/300VW-3D_closed_eyes', exist_ok=True)


    model = FaceModel()
    img_list = list(Path('300WLP').glob('**/*.jpg')) + \
                list(Path('GANnotation/300VW-3D_closed_eyes').glob('**/*.jpg'))
                
    img_list = [img_list[idx] for idx in range(len(img_list)) if idx%10==0]
    bag = []
    print(f'Push item to bag: ')
    for img_path in tqdm.tqdm(img_list):
        pts_path = str(img_path).replace('jpg', 'mat')
        bag.append((str(img_path), pts_path))
    
    for folder_path in glob.glob('GANnotation/300VW-3D_closed_eyes/*'):
        folder_img_name = folder_path.split('/')[-1]
        os.makedirs(
            os.path.join(f'squeezed_face_3ddfa/300VW-3D_closed_eyes', folder_img_name),
            exist_ok=True
        )

    def task(item, debug=False):
        img_path, pts_path = item
        name = img_path.split('/')[-1].split('.')[0]
        original_img = cv2.imread(img_path)
        height, width = original_img.shape[:2]
        original_pts = sio.loadmat(pts_path)['pt3d']

        if 'closed_eyes' in img_path:
            folder_id = img_path.split('/')[-2]
            full_folder_id = f'300VW-3D_closed_eyes/{folder_id}'
        elif 'opened_eyes' in img_path:
            folder_id = img_path.split('/')[-2]
            full_folder_id = f'300VW-3D_opened_eyes/{folder_id}'
        else:
            full_folder_id = '300WLP-verified'

        if debug:
            id = full_folder_id.replace('/','_')
            draw_landmarks(original_img.copy(), original_pts.copy(), f'debug/{id}_{name}_original.jpg')


        box_left, box_top, box_right, box_bot = get_landmarks_wrapbox(original_pts)
        size = min(box_right-box_left, box_bot-box_top)

        pad_ratio = random.uniform(0.1, 0.3)
        
        pad = int(size * pad_ratio)

        squeeze_type = random.choice(['v', 'h'])

        if squeeze_type == 'v' or squeeze_type == 'vertical':
            padded_img = cv2.copyMakeBorder(original_img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, 0)
            padded_pts = original_pts.T
            padded_pts[0] += pad
            padded_pts = padded_pts.T
        elif squeeze_type == 'h' or squeeze_type == 'horizontal':
            padded_img = cv2.copyMakeBorder(original_img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, 0)
            padded_pts = original_pts.T
            padded_pts[1] += pad
            padded_pts = padded_pts.T

        resized_img, resized_pts = resize_face_landmarks(padded_img, padded_pts, shape=(width, height))

        gen_img, gen_info = model.generate_3ddfa_params(resized_img, resized_pts, False, expand_ratio=1.)

        img_out_path = os.path.join(f'squeezed_face_3ddfa/{full_folder_id}', f'{name}.jpg')
        params_out_path = os.path.join(f'squeezed_face_3ddfa/{full_folder_id}', f'{name}.mat')
        cv2.imwrite(img_out_path, gen_img)
        sio.savemat(params_out_path, gen_info)

        if debug:
            re_pts = fm.reconstruct_vertex(gen_img, gen_info['params'], False)[:,:2][fm.bfm.kpt_ind]
            draw_landmarks(gen_img, re_pts, f'debug/{id}_{name}.jpg')


    debug = True
    shutil.rmtree('debug', ignore_errors=True)
    os.makedirs('debug', exist_ok=True)
    for idx in tqdm.tqdm(range(len(bag)), total=len(bag)):
        if idx % 300 == 0:
            debug = True
            task(bag[idx], debug)
            debug = False
        else:
            task(bag[idx], debug)