import cv2
from cv2 import data
from numpy.lib.arraysetops import isin
from numpy.lib.function_base import extract
import torch
import numpy as np
import scipy.io as sio
import os
from model import Generator
import utils
import multiprocessing as mp
import tqdm
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
# mpl = mp.log_to_stderr()
# mpl.setLevel(logging.INFO)

def preprocess_closed_eyes_single_v1(item):
    img, pts_2d, pts_3d = item

    pts_2d = utils.close_eyes_68_ver_1(pts_2d)
    pts_2d = utils.extract_66_from_68(pts_2d)

    # img, pts = utils.crop_face_landmarks(img, pts)
    # img, pts = utils.resize_face_landmarks(img, pts, (128, 128))

    img = (img/255.).transpose(2,0,1).astype(float)

    frame_w = int(np.floor(2*pts_2d.max()))
    frame = np.zeros((frame_w,frame_w,3))
    pts_2d, pts_3d = utils.reduced_crop(frame, pts_2d, pts_3d, size=128, tight=16)
    pts_3d = utils.replace_eyes(pts_2d, pts_3d)

    A_to_B = utils.generate_Ginput(img, pts_2d, sigma=1, size=128)

    return np.expand_dims(A_to_B,0), pts_3d


def preprocess_closed_eyes_single_v2(item):
    img, pts_2d, pts_3d = item

    pts_2d = utils.close_eyes_68_ver_2(pts_2d)
    pts_2d = utils.extract_66_from_68(pts_2d)

    # img, pts = utils.crop_face_landmarks(img, pts)
    # img, pts = utils.resize_face_landmarks(img, pts, (128, 128))

    img = (img/255.).transpose(2,0,1).astype(float)

    frame_w = int(np.floor(2*pts_2d.max()))
    frame = np.zeros((frame_w,frame_w,3))
    pts_2d, pts_3d = utils.reduced_crop(frame, pts_2d, pts_3d, size=128, tight=16)
    pts_3d = utils.replace_eyes(pts_2d, pts_3d)

    A_to_B = utils.generate_Ginput(img, pts_2d, sigma=1, size=128)

    return np.expand_dims(A_to_B,0), pts_3d


class GANnotation:
    def __init__(self, path_to_model='',enable_cuda=True, train=False):
        self.GEN = Generator()
        self.enable_cuda = enable_cuda
        self.size = 128
        self.tight = 16
        net_weigths = torch.load(path_to_model,map_location=lambda storage, loc: storage)
        net_dict = {k.replace('module.',''): v for k, v in net_weigths['state_dict'].items()}
        self.GEN.load_state_dict(net_dict)
        if self.enable_cuda:
            self.GEN = self.GEN.cuda()
        self.GEN.eval()
        
    def reenactment(self,image,videocoords):
        #image, points = utils.process_image(image,coords,angle=0, flip=False, sigma=1,size=128, tight=16) # do this outside
        frame_w = int(np.floor(2*videocoords.max()))
        frame = np.zeros((frame_w,frame_w,3))
        if videocoords.ndim == 2:
            videocoords = videocoords.reshape((66,2,1))
        n_frames = videocoords.shape[2]
        cropped_points = np.zeros((66,2,n_frames))
        images = []
        for i in range(0,n_frames):
            print(i)
            if videocoords[0,0,i] > 0:
                target = videocoords[:,:,i]
                _, target = utils.crop( frame , target, size=128, tight=16 )
                cropped_points[:,:,i] = target
                target = utils.close_eyes_68_ver_2(target)
                A_to_B = np.expand_dims(utils.generate_Ginput(image,target,sigma=1,size=128), 0)

                if self.enable_cuda:
                    A_to_B = torch.tensor(A_to_B, dtype=torch.float).cuda()
                else:
                    A_to_B = torch.tensor(A_to_B, dtype=torch.float)

                generated = 0.5*(self.GEN(torch.autograd.Variable(A_to_B)).data[0,:,:,:].cpu().numpy().swapaxes(0,1).swapaxes(1,2) + 1)
                imout = (255*generated).astype('uint8')
                imout = np.ascontiguousarray(imout, dtype=np.uint8)

                # for pt in target:
                #     pt = pt.astype(np.uint8)
                #     imout = cv2.circle(imout, (pt[0], pt[1]), 3, (0,255,0), -1, 8)

                images.append(imout)
        return images, cropped_points

    def _preprocess_opened_eyes_single(self, item):
        img, pts_2d, pts_3d = item

        pts_2d = utils.open_eyes_68(pts_2d)
        pts_2d = utils.extract_66_from_68(pts_2d)

        # img, pts = utils.crop_face_landmarks(img, pts)
        # img, pts = utils.resize_face_landmarks(img, pts, (128, 128))

        img = (img/255.).transpose(2,0,1).astype(float)

        frame_w = int(np.floor(2*pts_2d.max()))
        frame = np.zeros((frame_w,frame_w,3))
        pts_2d, pts_3d = utils.reduced_crop(frame, pts_2d, pts_3d, size=128, tight=16)
        pts_3d = utils.replace_eyes(pts_2d, pts_3d)

        A_to_B = utils.generate_Ginput(img, pts_2d, sigma=1, size=128)

        return np.expand_dims(A_to_B,0), pts_3d

    @torch.no_grad()
    def gen_close_eyes_batch(self, img_list, pts_list, out_dir='300VW-3D_cropped_closed_eyes', batch_size=64, closed_eyes=True, version=1):
        os.makedirs(out_dir, exist_ok=True)
        bag = []
        
        logging.info('=> Load image and landmarks: ')
        for idx in tqdm.tqdm(range(len(img_list))):
            img = img_list[idx]
            pts = pts_list[idx]
            if isinstance(img, str):
                img = cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2RGB)
            
            if isinstance(pts, str):
                pts_2d = utils.read_pts(pts)['pt2d']
                pts_3d = utils.read_pts(pts)['pt3d']

            bag.append((img, pts_2d, pts_3d))
        
        logging.info('=> Preprocess batch: ')
        if version == 1:
            with mp.Pool(mp.cpu_count()) as p:
                data_batch = list(tqdm.tqdm(p.imap(
                        preprocess_closed_eyes_single_v1, bag
                    ), total=len(bag)))
        elif version == 2:
            with mp.Pool(mp.cpu_count()) as p:
                data_batch = list(tqdm.tqdm(p.imap(
                        preprocess_closed_eyes_single_v2, bag
                    ), total=len(bag)))
        else:
            return

        image_batch = [data_batch[i][0] for i in range(len(data_batch))]
        pts_batch = [data_batch[i][1] for i in range(len(data_batch))]
        image_batch = np.concatenate(image_batch, 0)

        if self.enable_cuda:
            image_batch = torch.tensor(image_batch, device='cuda', dtype=torch.float)
        else:
            image_batch = torch.tensor(image_batch, device='cpu', dtype=torch.float)

        div_batch = len(image_batch) // batch_size
        mod_batch = len(image_batch) % batch_size
        all_result = []

        logging.info('=> Predict batch: ')
        for index in tqdm.tqdm(range(div_batch)):
            sub_batch_tensor = image_batch[index*batch_size:
                                            (index+1)*batch_size, :, :, :]
            results = 0.5*(self.GEN(torch.autograd.Variable(sub_batch_tensor)).cpu().numpy().transpose(0,2,3,1) + 1)
            all_result.extend(results)

        if mod_batch > 0:
            sub_batch_tensor = image_batch[-mod_batch:, :, :, :]
            results = 0.5*(self.GEN(torch.autograd.Variable(sub_batch_tensor)).cpu().numpy().transpose(0,2,3,1) + 1)

            all_result.extend(results)
        
        all_result = (255*np.array(all_result)).astype('uint8')
        all_result = np.ascontiguousarray(all_result, dtype=np.uint8)

        print('=> Saving image: ')
        for idx in tqdm.tqdm(range(len(all_result)), total=len(all_result)):
            folder_id = img_list[idx].split('/')[-2]
            img_name = img_list[idx].split('/')[-1]
            # os.makedirs(os.path.join(out_dir, folder_id), exist_ok=True)
            new_image = all_result[idx]
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

            # pts = utils.read_pts(pts_list[idx])['pt3d']
            pts = pts_batch[idx]

            image_save_path = os.path.join(out_dir, folder_id, img_name)
            cv2.imwrite(image_save_path, new_image)

            pts_save_path = os.path.join(out_dir, folder_id, img_name.replace('jpg', 'mat'))
            sio.savemat(pts_save_path, {'pt3d': pts})
            
            # box_left = int(np.ceil(np.min(pts.T[0])))
            # box_right = int(np.ceil(np.max(pts.T[0])))
            # box_top = int(np.ceil(np.min(pts.T[1])))
            # box_bot = int(np.ceil(np.max(pts.T[1])))
            # print([int(np.ceil((box_left+box_right)/2)), int(np.ceil((box_top+box_bot)/2))])
            # for i in range(pts.shape[0]):
            #     _pts = pts[i].astype(int)
            #     _img = cv2.circle(new_image, (_pts[0], _pts[1]),2,(0,255,0), -1, 5)
            
            # cv2.imwrite(f'test_images/{img_name}', _img)




    
        


 
    











