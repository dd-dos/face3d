import os
import sys
import time
import ipdb
from sympy.testing.runtests import SymPyTestResults
import torch
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numba

def process_image(image,points,angle=0, flip=False, sigma=1,size=128, tight=16):
    if angle > 0:
        if np.random.rand(1) > 0.4:
            tmp_angle = np.random.randn(1) * angle
            image,points = affine_trans(image,points, tmp_angle)
    image, points = crop( image , points, size, tight )
    if flip:
        if np.random.rand(1) > 0.5:
            image,points = flip_ImAndPts(image,points)
        
    image = image/255.0
    image = torch.from_numpy(image.swapaxes(2,1).swapaxes(1,0))
    image = image.type_as(torch.FloatTensor())

    source_maps = generate_maps(points, sigma, size)
    source_maps = source_maps.type_as(torch.FloatTensor())   

    return image, source_maps, points


def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss


def draw_gaussian(image, point, sigma):
    # Check if the gaussian is inside
    point[0] = round( point[0], 2)
    point[1] = round( point[1], 2)

    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] >
            image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) - int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) - int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
          ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1
    return image


def generate_maps(points, sigma, size=256):
    maps = None 
    for i in range(0,66):

        tpt = np.array([points[i,0],points[i,1]])
 
        map = draw_gaussian(np.zeros((size,size)),tpt,sigma=sigma)
 
        if maps is None:
            # maps = torch.from_numpy(map).unsqueeze(0)
            maps = np.expand_dims(map, 0)
        else:
            # maps = np.cat((maps, torch.from_numpy(map).unsqueeze(0)), 0) 
            maps = np.concatenate((maps, np.expand_dims(map, 0)), 0)

    return maps 


def crop( image, landmarks , size, tight=8):
    delta_x = np.max(landmarks[:,0]) - np.min(landmarks[:,0])
    delta_y = np.max(landmarks[:,1]) - np.min(landmarks[:,1])
    delta = 0.5*(delta_x + delta_y)
    if delta < 20:
        tight_aux = 8
    else:
        tight_aux = int(tight * delta/100)
    pts_ = landmarks.copy()
    w = image.shape[1]
    h = image.shape[0]
    min_x = int(np.maximum( np.round( np.min(landmarks[:,0]) ) - tight_aux , 0 ))
    min_y = int(np.maximum( np.round( np.min(landmarks[:,1]) ) - tight_aux , 0 ))
    max_x = int(np.minimum( np.round( np.max(landmarks[:,0]) ) + tight_aux , w-1 ))
    max_y = int(np.minimum( np.round( np.max(landmarks[:,1]) ) + tight_aux , h-1 ))
    image = image[min_y:max_y,min_x:max_x,:]
    pts_[:,0] = pts_[:,0] - min_x
    pts_[:,1] = pts_[:,1] - min_y
    sw = size/image.shape[1]
    sh = size/image.shape[0]
    im = cv2.resize(image, dsize=(size,size),
                    interpolation=cv2.INTER_LINEAR)
    
    pts_[:,0] = pts_[:,0]*sw
    pts_[:,1] = pts_[:,1]*sh
    return im, pts_

@numba.njit()
def reduced_crop(image, landmarks , size, tight=8):
    delta_x = np.max(landmarks[:,0]) - np.min(landmarks[:,0])
    delta_y = np.max(landmarks[:,1]) - np.min(landmarks[:,1])
    delta = 0.5*(delta_x + delta_y)
    if delta < 20:
        tight_aux = 8
    else:
        tight_aux = int(tight * delta/100)
    pts_ = landmarks
    w = image.shape[1]
    h = image.shape[0]
    min_x = int(np.maximum( np.round( np.min(landmarks[:,0]) ) - tight_aux , 0 ))
    min_y = int(np.maximum( np.round( np.min(landmarks[:,1]) ) - tight_aux , 0 ))
    max_x = int(np.minimum( np.round( np.max(landmarks[:,0]) ) + tight_aux , w-1 ))
    max_y = int(np.minimum( np.round( np.max(landmarks[:,1]) ) + tight_aux , h-1 ))
    image = image[min_y:max_y,min_x:max_x,:]
    pts_[:,0] = pts_[:,0] - min_x
    pts_[:,1] = pts_[:,1] - min_y
    sw = size/image.shape[1]
    sh = size/image.shape[0]
    
    pts_[:,0] = pts_[:,0]*sw
    pts_[:,1] = pts_[:,1]*sh
    return pts_


def generate_Ginput( img, target_pts , sigma , size=256 ):
    target_maps = generate_maps(target_pts, sigma, size).astype(float)
    # target_maps = target_maps.type_as(torch.FloatTensor)

    A_to_B = np.concatenate((img, target_maps),0)
    return A_to_B


def flip_ImAndPts(image,landmarks):
    flipImg = cv2.flip(image, 1)
    pts_mirror = np.hstack(([range(17,0,-1), range(27,17,-1), range(28,32,1), range(36,31,-1), range(46,42,-1),48,47, range(40,36,-1),42,41,range(55,48,-1),range(60,55,-1),range(63,60,-1),range(66,63,-1)]))
    pts_mirror = pts_mirror - 1
    flipLnd = np.copy(landmarks)
    flipLnd[:,0] = image.shape[1] - landmarks[pts_mirror,0]
    flipLnd[:,1] = landmarks[pts_mirror,1]
    return flipImg,flipLnd


def fliplr_face_landmarks(img, pts, reverse=True):
    """
    Flip left right image and landmarks.

    Args:
        :img: input image.
        :pts: np.array landmarks of shape (68,-1) 
    """

    '''
    Return points to orginal status.
    '''
    width, height = img.shape[:2]
    if reverse:
        pts.T[0] += width/2
        pts.T[1] += height/2
        pts.T[1] = height - 1 - pts.T[1]

    pts.T[0] = width - pts.T[0]

    matchedParts = ([0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
                    [17, 26], [18, 25], [19, 24], [20, 23], [21, 22], [36, 45], [37, 44],
                    [38, 43], [39, 42], [41, 46], [40, 47], [31, 35], [32, 34], [50, 52],
                    [49, 53], [48, 54], [61, 63], [67, 65], [59, 55], [58, 56], [60,64])

    # Change left-right parts
    for pair in matchedParts:
        pts[pair] = pts[[pair[1], pair[0]]]

    return cv2.flip(img, 1), pts


def affine_trans(image,landmarks,angle=None):
    if angle is None:
        angle = 30*torch.randn(1)
       
    
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    dst = cv2.warpAffine(image, M, (nW,nH))
    new_landmarks = np.concatenate((landmarks,np.ones((66,1))),axis=1)
    new_landmarks = new_landmarks.dot(M.transpose())

    return dst, new_landmarks


def gram_matrix(input):
    bsize, ch, r, c = input.size()  
    features = input.view(bsize * ch, r * c) 
    G = torch.mm(features, features.t())  
    return G.div(bsize*ch*r*c)


def show_pts(img, pts):
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if np.mean(img) <= 1:
        img = img*255
    
    if img.shape[0] == 3:
        img = img.transpose(1,2,0)

    img = np.ascontiguousarray(img, dtype=np.uint8)
    _img = img.copy()

    try:
        for i in range(pts.shape[0]):
            _pts = pts[i].astype(int)
            _img = cv2.circle(_img, (_pts[0], _pts[1]),2,(0,255,0), -1, 8)
    except Exception as e:
        print(e)
        import ipdb; ipdb.set_trace(context=10)
    
    # _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    # Image.fromarray(_img).show()
    cv2.imshow('', _img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_vertices(vertices: np.ndarray, v_type='3D'):
    if v_type=='3D':
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        _vertices = vertices.transpose(1, 0)
        ax.scatter(_vertices[0],
                   _vertices[1],
                   _vertices[2],
                   marker=".")
        
        # ax.axis('off')
        plt.show() 
        plt.close()
    elif v_type=='2D':
        # ax, fig = plt.figure()

        # _vertices = vertices.transpose(1, 0)
        # ax.scatter(_vertices[0],
        #            _vertices[1],
        #            marker=".")
        _vertices = vertices.transpose(1, 0)
        plt.scatter(_vertices[0],
                    _vertices[1],
                    marker='.')
        
        # plt.axis('off')
        plt.show() 
        plt.close()        
    else:
        return


def read_pts(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))


@numba.njit()
def extract_66_from_68(pts):
    return np.concatenate((pts[:60], pts[61:64], pts[65:]), axis=0)


@numba.njit()
def crop_multi_face_landmarks(img, pts_2d, landmarks, expand_ratio=1.0):
    """
    Pad and crop to retain landmarks when rotating.

    Params:
    :img: image to pad.
    :landmarks: 68 landmarks points.
    """
    # Get the box that wrap all landmarks.
    # box_top, box_left, box_bot, box_right = \
    # get_landmarks_wrapbox(landmarks)
    box_left = np.min(landmarks.T[0])
    box_right = np.max(landmarks.T[0])
    box_top = np.min(landmarks.T[1])
    box_bot = np.max(landmarks.T[1])

    # Crop image to get the largest square region that satisfied:
    # 1. Contains all landmarks
    # 2. Center of the landmarks box is the center of the region.
    center = [(box_left+box_right)/2, (box_top+box_bot)/2]
    
    # Get the diameter of largest region 
    # that a landmark can reach when rotating.
    box_height = box_bot-box_top
    box_width = box_right-box_left
    radius = max(box_height, box_width) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    # Crop a bit larger.
    max_length = np.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    crop_size = int(max_length/2 * expand_ratio)

    img_height, img_width, channel = img.shape
    canvas = np.zeros((img_height+2*crop_size, img_width+2*crop_size, channel), dtype=np.uint8)
    canvas[crop_size:img_height+crop_size, crop_size:img_width+crop_size, :] = img

    # Adjust center coord.
    center_x += crop_size
    center_y += crop_size

    # Top left bottom right.
    y1 = center_y-int(crop_size)
    x1 = center_x-int(crop_size)
    y2 = center_y+int(crop_size)
    x2 = center_x+int(crop_size)

    # Crop image.
    img = canvas[y1:y2, x1:x2]
    
    # Adjust landmarks and center
    landmarks.T[0] = landmarks.T[0] - x1 + crop_size
    landmarks.T[1] = landmarks.T[1] - y1 + crop_size

    pts_2d.T[0] += crop_size - x1
    pts_2d.T[1] += crop_size - y1
 
    return img, pts_2d, landmarks

import sympy
def close_eyes_68_ver_1(pts):
    '''
    Simple version.
    '''
    # pts[37] = pts[41]
    # pts[38] = pts[40]
    # pts[43] = pts[47]
    # pts[44] = pts[46]

    '''
    More complex version.
    '''
    pt_37 = sympy.Point(pts[37])
    pt_38 = sympy.Point(pts[38])
    pt_43 = sympy.Point(pts[43])
    pt_44 = sympy.Point(pts[44])

    pt_36 = sympy.Point(pts[36])
    pt_39 = sympy.Point(pts[39])
    right_axis = sympy.geometry.Line(pt_36, pt_39)

    pt_42 = sympy.Point(pts[42])
    pt_45 = sympy.Point(pts[45])
    left_axis = sympy.geometry.Line(pt_42, pt_45)

    projected_pt_37 = right_axis.projection(pt_37)
    projected_pt_38 = right_axis.projection(pt_38)

    projected_pt_43 = left_axis.projection(pt_43)
    projected_pt_44 = left_axis.projection(pt_44)

    pts[41] = pts[37] = np.array(projected_pt_37).astype(np.float32)
    pts[40] = pts[38] = np.array(projected_pt_38).astype(np.float32)
    pts[47] = pts[43] = np.array(projected_pt_43).astype(np.float32)
    pts[46] = pts[44] = np.array(projected_pt_44).astype(np.float32)
    
    '''
    Move lower eyes up.
    '''
    # pt_41 = sympy.Point(pts[41])
    # pt_40 = sympy.Point(pts[40])
    # pt_47 = sympy.Point(pts[47])
    # pt_46 = sympy.Point(pts[46])

    # projected_pt_41 = right_axis.projection(pt_41)
    # projected_pt_40 = right_axis.projection(pt_40)

    # projected_pt_47 = left_axis.projection(pt_47)
    # projected_pt_46 = left_axis.projection(pt_46)

    # pts[41] = np.array(projected_pt_37).astype(np.float32)
    # pts[40] = np.array(projected_pt_38).astype(np.float32)
    # pts[47] = np.array(projected_pt_43).astype(np.float32)
    # pts[46] = np.array(projected_pt_44).astype(np.float32)

    return pts

def close_eyes_68_ver_2(pts):
    '''
    Simple version.
    '''
    pts[37] = pts[41] = pts[37] + (pts[41] - pts[37])*3/4
    pts[38] = pts[40] = pts[38] + (pts[40] - pts[38])*3/4
    pts[43] = pts[47] = pts[43] + (pts[47] - pts[43])*3/4
    pts[44] = pts[46] = pts[44] + (pts[46] - pts[44])*3/4

    return pts

def resize_face_landmarks(img, pts_2d, landmarks, shape=(256,256)):
    height, width, _ = img.shape

    width_ratio = shape[0] / width
    height_ratio = shape[1] / height

    img = cv2.resize(img, shape)

    # landmarks.T[0] = landmarks.T[0]*width_ratio
    # landmarks.T[1] = landmarks.T[1]*height_ratio

    landmarks *= np.array([width_ratio, height_ratio])
    pts_2d *= np.array([width_ratio, height_ratio])

    return img, pts_2d, landmarks

def get_eyes(pts):
    right = pts[36:42]
    left = pts[42:48]

    return {
        'left': left,
        'right': right
    }

def replace_eyes(pts_2d, pts_3D):
    pts_3D[37] = pts_2d[37]
    pts_3D[41] = pts_2d[41]
    pts_3D[38] = pts_2d[38]
    pts_3D[40] = pts_2d[40]
    pts_3D[43] = pts_2d[43]
    pts_3D[47] = pts_2d[47]
    pts_3D[44] = pts_2d[44]
    pts_3D[46] = pts_2d[46]

    pts_3D[36] = pts_2d[36]
    pts_3D[39] = pts_2d[39]
    pts_3D[42] = pts_2d[42]
    pts_3D[45] = pts_2d[45]

    return pts_3D

def draw_pts(img, pts):
    foo_img = img.copy()
    for pt in pts:
        pt = pt.astype(int)
        foo_img = cv2.circle(foo_img, pt,2,(0,255,0), -1, 8)
    cv2.imwrite(f'foo.jpg', foo_img)