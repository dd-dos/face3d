import logging
from sys import version
import GANnotation
from PIL import Image
from pathlib import Path
import glob
import numpy as np
import tqdm



model = GANnotation.GANnotation(path_to_model='models/myGEN.pth', enable_cuda=True)

logging.info('=> Read data path...')
pts_list = np.array(list(Path('300VW-3D_cropped_semi_eyes').glob('**/*.mat'))).astype(str)

img_list = [pts_list[i].replace('mat', 'jpg') for i in range(len(pts_list))]
logging.info('=> Done reading.')

'''
In case you only want a part of the list.
'''
sub_pts_list = [pts_list[i] for i in range(len(pts_list)) if i%1==0]
sub_img_list = [img_list[i] for i in range(len(img_list)) if i%1==0]
assert len(sub_pts_list) == len(sub_img_list)
split_size=1000
div_batch = len(sub_pts_list) // split_size
mod_batch = len(sub_pts_list) % split_size

import shutil
import os
# shutil.rmtree('300VW-3D_cropped_closed_eyes_GAN', ignore_errors=True)
shutil.rmtree('test_images', ignore_errors=True)
os.makedirs('test_images', exist_ok=True)
os.makedirs('300VW-3D_cropped_closed_eyes_GAN', exist_ok=True)
os.makedirs('300VW-3D_cropped_closed_eyes_GAN/ver_1', exist_ok=True)
os.makedirs('300VW-3D_cropped_closed_eyes_GAN/ver_2', exist_ok=True)

for folder_path in glob.glob('300VW-3D_cropped_semi_eyes/*'):
    folder_name = folder_path.split('/')[-1]
    os.makedirs(
        os.path.join('300VW-3D_cropped_closed_eyes_GAN/ver_1', folder_name),
        exist_ok=True
    )
    os.makedirs(
        os.path.join('300VW-3D_cropped_closed_eyes_GAN/ver_2', folder_name),
        exist_ok=True
    )

# logging.info('Generating closed eyes version 1.')
# for idx in tqdm.tqdm(range(div_batch)):
#     model.gen_close_eyes_batch(
#         sub_img_list[idx*split_size:(idx+1)*split_size], 
#         sub_pts_list[idx*split_size:(idx+1)*split_size], 
#         batch_size=128, 
#         closed_eyes=True,
#         out_dir='300VW-3D_cropped_closed_eyes_GAN/ver_1'
#     )
# if mod_batch > 0:
#     model.gen_close_eyes_batch(
#         sub_img_list[-mod_batch:], 
#         sub_pts_list[-mod_batch:], 
#         batch_size=128, 
#         closed_eyes=True,
#         out_dir='300VW-3D_cropped_closed_eyes_GAN/ver_1'
#     )


logging.info('Generating closed eyes version 2.')
for idx in tqdm.tqdm(range(div_batch)):
    model.gen_close_eyes_batch(
        sub_img_list[idx*split_size:(idx+1)*split_size], 
        sub_pts_list[idx*split_size:(idx+1)*split_size], 
        batch_size=128, 
        closed_eyes=True,
        out_dir='300VW-3D_cropped_closed_eyes_GAN/ver_2',
        version=2
    )
if mod_batch > 0:
    model.gen_close_eyes_batch(
        sub_img_list[-mod_batch:], 
        sub_pts_list[-mod_batch:], 
        batch_size=128, 
        closed_eyes=True,
        out_dir='300VW-3D_cropped_closed_eyes_GAN/ver_2',
        version=2
    )
