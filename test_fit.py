import os
import multiprocessing as mp
from pathlib import Path
from face3d.face_model import FaceModel
from face3d.utils import show_pts
import tqdm
import cv2
import scipy.io as sio
import glob
import numpy as np


if __name__=='__main__':
    model = FaceModel(bfm_path='examples/Data/BFM/Out/BFM.mat')
    img_list = list(Path('300VW-3D_cropped').glob('**/*.jpg'))
    bag = []
    print(f'Push item to bag: ')
    for img_path in tqdm.tqdm(img_list):
        pts_path = str(img_path).replace('jpg', 'mat')
        bag.append((str(img_path), pts_path))

    def task(item):
        img_path, pts_path = item
        print(img_path)

        img = cv2.imread(img_path)
        pts = sio.loadmat(pts_path)['pt3d']
        img, params = model.generate_3ddfa_params(img, pts)
        import ipdb; ipdb.set_trace(context=10)
        vert = model.reconstruct_vertex(img, params)
        show_pts(img, vert[model.bfm.kpt_ind])

    # with mp.Pool(mp.cpu_count()-2) as p:
    #     r = list(tqdm.tqdm(p.imap(task, bag), total=len(bag)))
    for item in tqdm.tqdm(bag):
        task(item)