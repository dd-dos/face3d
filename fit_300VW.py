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
    os.makedirs('300VW-3D_cropped_3ddfa', exist_ok=True)
    for folder_path in glob.glob('300VW-3D_cropped/*'):
        folder_name = folder_path.split('/')[-1]
        os.makedirs(
            os.path.join('300VW-3D_cropped_3ddfa', folder_name),
            exist_ok=True
        )

    model = FaceModel(bfm_path='examples/Data/BFM/Out/BFM.mat')
    img_list = list(Path('300VW-3D_cropped').glob('**/*.jpg'))
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

        img_out_path = img_path.replace('300VW-3D_cropped','300VW-3D_cropped_3ddfa')
        params_out_path = img_path.replace('300VW-3D_cropped','300VW-3D_cropped_3ddfa').replace('jpg', 'npy')
        cv2.imwrite(img_out_path, img)
        np.save(params_out_path, params)

    with mp.Pool(6) as p:
        r = list(tqdm.tqdm(p.imap(task, bag), total=len(bag)))
