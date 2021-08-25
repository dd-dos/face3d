import os
import multiprocessing as mp
from pathlib import Path
from face3d.face_model import FaceModel
import tqdm
import cv2
import scipy.io as sio
import glob
import numpy as np
import utils


if __name__=='__main__':
    os.makedirs('300WLP_3ddfa', exist_ok=True)
    os.makedirs('300WLP_3ddfa/300WLP_3ddfa-verified', exist_ok=True)

    model = FaceModel()
    img_list = list(Path('300WLP').glob('**/*.jpg'))
    bag = []
    print(f'Push item to bag: ')
    for img_path in tqdm.tqdm(img_list):
        pts_path = str(img_path).replace('jpg', 'mat')
        bag.append((str(img_path), pts_path))

    def task(item):
        img_path, pts_path = item
        name = img_path.split('/')[-1].split('.')[0]
        img = cv2.imread(img_path)
        pts = sio.loadmat(pts_path)['pt3d']

        fliplr_img, fliplr_pts = utils.fliplr_face_landmarks(img, pts)

        img, params = model.generate_3ddfa_params(img, pts, expand_ratio=1.)
        img_out_path = os.path.join('300WLP_3ddfa/300WLP_3ddfa-verified', f'{name}.jpg')
        params_out_path = os.path.join('300WLP_3ddfa/300WLP_3ddfa-verified', f'{name}.mat')
        cv2.imwrite(img_out_path, img)
        sio.savemat(params_out_path, params)

        fliplr_img, fliplr_params = model.generate_3ddfa_params(fliplr_img, fliplr_pts, expand_ratio=1.)
        fliplr_img_out_path = os.path.join('300WLP_3ddfa/300WLP_3ddfa-verified', f'{name}_fliplr.jpg')
        fliplr_params_out_path = os.path.join('300WLP_3ddfa/300WLP_3ddfa-verified', f'{name}_fliplr.mat')
        cv2.imwrite(fliplr_img_out_path, fliplr_img)
        sio.savemat(fliplr_params_out_path, fliplr_params)

        # cv2.imshow('', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # with mp.Pool(mp.cpu_count()-2) as p:
    #     r = list(tqdm.tqdm(p.imap(task, bag), total=len(bag)))
    for item in tqdm.tqdm(bag):
        task(item)