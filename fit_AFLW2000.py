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

if __name__=='__main__':
    shutil.rmtree('AFLW2000_3ddfa', ignore_errors=True)
    os.makedirs('AFLW2000_3ddfa', exist_ok=True)

    model = FaceModel()
    img_list = list(Path('AFLW2000').glob('**/*.jpg'))
    bag = []
    print(f'Push item to bag: ')
    for img_path in tqdm.tqdm(img_list):
        pts_path = str(img_path).replace('jpg', 'mat')
        bag.append((str(img_path), pts_path))

    def task(item):
        img_path, pts_path = item
        img = cv2.imread(img_path)
        pts = sio.loadmat(pts_path)['pt3d_68'][:2].T
        img, params = model.generate_3ddfa_params(img, pts)

        img_out_path = img_path.replace('AFLW2000','AFLW2000_3ddfa')
        params_out_path = img_path.replace('AFLW2000','AFLW2000_3ddfa').replace('jpg', 'mat')
        cv2.imwrite(img_out_path, img)
        sio.savemat(params_out_path, params)

        # cv2.imshow('', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # with mp.Pool(mp.cpu_count()-2) as p:
    #     r = list(tqdm.tqdm(p.imap(task, bag), total=len(bag)))
    for item in tqdm.tqdm(bag):
        task(item)