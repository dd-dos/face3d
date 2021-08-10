import os
import multiprocessing as mp
from pathlib import Path
from face3d.face_model import FaceModel
import tqdm
import cv2
import scipy.io as sio
import glob
import numpy as np


if __name__=='__main__':
    os.makedirs('300WLP_3ddfa', exist_ok=True)
    os.makedirs('300WLP_3ddfa/300WLP_3ddfa-verified', exist_ok=True)

    model = FaceModel(bfm_path='examples/Data/BFM/Out/BFM.mat')
    img_list = list(Path('300WLP').glob('**/*.jpg'))
    bag = []
    print(f'Push item to bag: ')
    for img_path in tqdm.tqdm(img_list):
        pts_path = str(img_path).replace('jpg', 'mat')
        bag.append((str(img_path), pts_path))

    def task(item):
        img_path, pts_path = item
        img = cv2.imread(img_path)
        pts = sio.loadmat(pts_path)['pt3d']
        img, params = model.generate_3ddfa_params(img, pts)

        img_out_path = img_path.replace('300WLP','300WLP_3ddfa')
        params_out_path = img_path.replace('300WLP','300WLP_3ddfa').replace('jpg', 'npy')
        cv2.imwrite(img_out_path, img)
        np.save(params_out_path, params)

    with mp.Pool(mp.cpu_count()) as p:
        r = list(tqdm.tqdm(p.imap(task, bag), total=len(bag)))