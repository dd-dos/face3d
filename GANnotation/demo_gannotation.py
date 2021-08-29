from re import I
import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import resize
import torch

import GANnotation
import utils
from utils import (close_eyes_68, crop_face_landmarks, extract_66_from_68,
                   read_pts, resize_face_landmarks, show_pts, show_vertices,
                   )

# image = cv2.cvtColor(cv2.imread('test_images/300VW_Dataset_2015_12_14/001/image/000001.jpg'),cv2.COLOR_BGR2RGB)
# points = read_pts('test_images/300VW_Dataset_2015_12_14/001/annot/000001.pts')
# points = close_eyes_68(points)
# points = extract_66_from_68(points)
# image, points = crop_face_landmarks(image, points)
# image, points = resize_face_landmarks(image, points, (128,128))
# points = points.reshape((66,2,1))

myGAN = GANnotation.GANnotation(path_to_model='models/myGEN.pth', enable_cuda=False)
points = np.loadtxt('test_images/test_1.txt').transpose().reshape(66,2,-1)[:,:,:1000]
image = cv2.cvtColor(cv2.imread('test_images/test_1.jpg'),cv2.COLOR_BGR2RGB)
# image = cv2.resize(image, (128, 128))

image = image/255.0
image = torch.from_numpy(image.swapaxes(2,1).swapaxes(1,0))
image = image.type_as(torch.FloatTensor())

images, cropped_pts = myGAN.reenactment(image,points)

# import ipdb; ipdb.set_trace(context=10)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test_1.avi',fourcc, 30.0, (128,128))
for imout in images:
    out.write(cv2.cvtColor(imout, cv2.COLOR_RGB2BGR))

out.release()
np.savetxt('test_1_cropped.txt', cropped_pts.reshape((132,-1)).transpose())
