import os
import multiprocessing as mp
from pathlib import Path
from face3d.face_model import FaceModel
from face3d.utils import check_frontal_face, isgray
import tqdm
import cv2
import scipy.io as sio
import glob
import numpy as np
import shutil

if __name__=='__main__':
    shutil.rmtree('AFLW2000_rotated_3ddfa', ignore_errors=True)
    os.makedirs('AFLW2000_rotated_3ddfa', exist_ok=True)

    model = FaceModel(bfm_path='examples/Data/BFM/Out/BFM.mat')
    img_list = list(Path('AFLW2000').glob('**/*.jpg'))
    bag = []
    print(f'Push item to bag: ')
    for img_path in tqdm.tqdm(img_list):
        pts_path = str(img_path).replace('jpg', 'mat')
        bag.append((str(img_path), pts_path))

    def task(item):
        img_path, pts_path = item
        img = cv2.imread(img_path)

        if isgray(img):
            return

        pts = sio.loadmat(pts_path)['pt3d_68'][:2].T
        if not check_frontal_face(pts):
            return
        
        img, params = model.generate_rotated_3d_img(img, pts)

        img_out_path = img_path.replace('AFLW2000','AFLW2000_rotated_3ddfa')
        params_out_path = img_path.replace('AFLW2000','AFLW2000_rotated_3ddfa').replace('jpg', 'npy')
        cv2.imwrite(img_out_path, img)
        np.save(params_out_path, params)

    with mp.Pool(mp.cpu_count()) as p:
        r = list(tqdm.tqdm(p.imap(task, bag), total=len(bag)))
