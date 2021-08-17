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
    shutil.rmtree('300VW-3D_cropped_non_closed_eyes_3ddfa', ignore_errors=True)
    os.makedirs('300VW-3D_cropped_non_closed_eyes_3ddfa', exist_ok=True)
    for folder_path in glob.glob('300VW-3D_cropped_non_closed_eyes/*'):
        folder_name = folder_path.split('/')[-1]
        os.makedirs(
            os.path.join('300VW-3D_cropped_non_closed_eyes_3ddfa', folder_name),
            exist_ok=True
        )

    model = FaceModel()
    img_list = list(Path('300VW-3D_cropped_non_closed_eyes').glob('**/*.jpg'))

    bag = []
    print(f'Push item to bag: ')
    for idx in tqdm.tqdm(range(len(img_list)), total=len(img_list)):
        if idx%4==0:
            img_path = img_list[idx]
            pts_path = str(img_path).replace('jpg', 'mat')
            bag.append((str(img_path), pts_path))

    def task(item):
        img_path, pts_path = item
        img = cv2.imread(img_path)
        pts = sio.loadmat(pts_path)['pt3d']
        img, params = model.generate_3ddfa_params(img, pts, preprocess=False)

        img_out_path = img_path.replace('300VW-3D_cropped_non_closed_eyes','300VW-3D_cropped_non_closed_eyes_3ddfa')
        params_out_path = img_out_path.replace('jpg', 'mat')
        cv2.imwrite(img_out_path, img)
        sio.savemat(params_out_path, params)

        # cv2.imshow('', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # with mp.Pool(2) as p:
    #     r = list(tqdm.tqdm(p.imap(task, bag), total=len(bag)))

    for item in tqdm.tqdm(bag):
        task(item)