import logging
from numpy.core.fromnumeric import squeeze
from numpy.random import rand
from face3d.mesh import light
import re
import cv2
import scipy.io as sio
from face3d.utils import *
from face3d.face_model import FaceModel, _get_colors
from face3d import mesh
from utils import close_eyes_68_ver_1, close_eyes_68_ver_2, crop
import numpy as np
import random
import time
fm = FaceModel()
import numba

def light_test(vertices, light_positions, light_intensities, h = 256, w = 256, colors=None, light=True):
    if colors is None:
        colors = fm.bfm.generate_colors(fm.bfm.get_tex_para())
        colors = colors/np.max(colors)

    if light == True:   
        lit_colors = mesh.light.add_light(vertices, fm.bfm.triangles, colors, light_positions, light_intensities)
    else:
        lit_colors = colors

    image_vertices = mesh.transform.to_image(vertices, h, w)
    # image_vertices = vertices
    rendering = mesh.render.render_colors(image_vertices, fm.bfm.triangles, lit_colors, h, w)
    rendering = np.minimum((np.maximum(rendering, 0)), 1)
    return rendering, image_vertices

def squeeze_face(img, pts, pad_ratio=None, squeeze_type='v'):
    ### Squeeze image and landmarks ###

    # height, width = img.shape[:2]
    box_left, box_top, box_right, box_bot = get_landmarks_wrapbox(pts)
    size = min(box_right-box_left, box_bot-box_top)

    if pad_ratio is None:
        pad_ratio = random.uniform(0.45, 0.55)
    
    pad = int(size * pad_ratio)

    if squeeze_type is None:
        squeeze_type = random.choice(['v', 'h'])

    if squeeze_type == 'v' or squeeze_type == 'vertical':
        padded_img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, 0)
        padded_pts = pts.T
        padded_pts[0] += pad
        padded_pts = padded_pts.T
    elif squeeze_type == 'h' or squeeze_type == 'horizontal':
        padded_img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, 0)
        padded_pts = pts.T
        padded_pts[1] += pad
        padded_pts = padded_pts.T

    resized_img, resized_pts = resize_face_landmarks(padded_img, padded_pts, shape=(width, height))
    
    ### Generate 3ddfa params ###
    n_img, info = fm.generate_3ddfa_params(resized_img, resized_pts, False, shape=(450,450))
    re_pts = fm.reconstruct_vertex(n_img, info['params'], False)[fm.bfm.kpt_ind]

    show_pts(n_img, re_pts)
    return n_img, info

@numba.njit()
def random_crop_substep(img, roi_box, params, expand_ratio=None, target_size=128):
    camera_matrix = params[:12].reshape(3, -1)

    trans = camera_matrix[:, 3]
    R1 = camera_matrix[0:1, :3]
    R2 = camera_matrix[1:2, :3]
    scale = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    rotation_matrix = np.concatenate((r1, r2, r3), 0)

    # Get the box that wrap all landmarks.
    box_left = roi_box[0]
    box_right = roi_box[2]
    box_top = roi_box[1]
    box_bot = roi_box[3]

    # Crop image to get the largest square region that satisfied:
    # 1. Contains all landmarks
    # 2. Center of the landmarks box is the center of the region.
    center = [(box_right+box_left)/2, (box_bot+box_top)/2]
    
    # Get the diameter of largest region 
    # that a landmark can reach when rotating.
    box_height = box_bot-box_top
    box_width = box_right-box_left
    radius = max(box_height, box_width) / 2

    max_length = 2*np.sqrt(2)*radius

    # Crop a bit larger.
    if expand_ratio is None:
        expand_ratio = random.uniform(1 / np.sqrt(2), 1.)
    elif expand_ratio < 1 / np.sqrt(2):
        '''
        Expand ratio is a little too big, 
        crop size will be negative and I don't wanna use more ops.")
        '''
        expand_ratio = 0.

    crop_size = int(max_length/2 * expand_ratio)

    img_height, img_width, channel = img.shape
    canvas = np.zeros((img_height+2*crop_size, img_width+2*crop_size, channel), dtype=np.uint8)
    canvas[crop_size:img_height+crop_size, crop_size:img_width+crop_size, :] = img

    shift_value = crop_size - radius
    '''
    0.125 is purely selected from visualization.
    '''
    shift_value_x = int(box_width * 0.125 + shift_value)
    shift_value_y = int(box_height * 0.125 + shift_value)

    shift_x = np.random.randint(-shift_value_x, shift_value_x)
    shift_y = np.random.randint(-shift_value_y, shift_value_y)

    # shift_x = shift_value_x
    # shift_y = shift_value_y

    center_x = int(center[0] + crop_size) + shift_x
    center_y = int(center[1] + crop_size) + shift_y

    # Top left bottom right.
    y1 = center_y-crop_size
    x1 = center_x-crop_size
    y2 = center_y+crop_size
    x2 = center_x+crop_size

    cropped_img = canvas[y1:y2, x1:x2]

    flip_matrix = np.array([[1,0,0],[0,-1,0],[0,0,1]], dtype=np.float64)
    flip_offset = np.array([0, img_height, 0], dtype=np.float64)
    norm_trans = np.array([img_width/2, img_height/2, 0], dtype=np.float64)

    cropped_trans = (flip_offset + np.array([-x1, -y1, 0])).reshape(3,) @ flip_matrix.T + norm_trans + trans

    resized_scale = scale / (2*crop_size) * target_size
    resized_trans = cropped_trans / (2*crop_size) * target_size

    re_scaled_rot_matrix = resized_scale * rotation_matrix
    re_camera_matrix = np.concatenate((re_scaled_rot_matrix, resized_trans.reshape(-1,1)), axis=1)
    re_params = np.concatenate((re_camera_matrix.reshape(12,1), params[12:].reshape(-1,1)), axis=0)

    return cropped_img, re_params

def random_crop(img, roi_box, params, expand_ratio=1, target_size=128):
    cropped_img, re_params = random_crop_substep(img, roi_box, params, expand_ratio, target_size)
    re_img = cv2.resize(cropped_img, (target_size, target_size))

    return re_img, re_params

@numba.njit()
def flip_substep(img, params):
    img_height, img_width = img.shape[:2]

    camera_matrix = params[:12].reshape(3, -1)

    trans = camera_matrix[:, 3]
    R1 = camera_matrix[0:1, :3]
    R2 = camera_matrix[1:2, :3]
    scale = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    rotation_matrix = np.concatenate((r1, r2, r3), 0)

    flip_matrix = np.array([[1,0,0],[0,-1,0],[0,0,1]], dtype=np.float64)
    flip_offset = np.array([0, img_height, 0], dtype=np.float64)

    norm_trans = np.array([img_width/2, img_height/2, 0], dtype=np.float64)

    flipped_rotation_matrix = flip_matrix @ (scale*rotation_matrix)
    flipped_trans = (trans + norm_trans) @ flip_matrix.T + flip_offset - norm_trans

    flipped_camera_matrix = np.concatenate((flipped_rotation_matrix, flipped_trans.reshape(-1,1)), axis=1)
    
    flipped_params = np.concatenate((flipped_camera_matrix.reshape(12,1), params[12:].reshape(-1,1)), axis=0)

    return flipped_params

def flip(img, params):
    flipped_params = flip_substep(img, params)
    flipped_img = cv2.flip(img, 0)

    re_pts = fm.reconstruct_vertex(flipped_img, flipped_params, False)[fm.bfm.kpt_ind]
    show_pts(flipped_img, re_pts)

if __name__=='__main__':
    # img = cv2.imread('examples/Data/image00050.jpg')
    # pts = sio.loadmat('examples/Data/image00050.mat')['pt3d_68'].T[:,:2]

    img = cv2.imread('examples/Data/300WLP-std_134212_1_0.jpg')
    pts = sio.loadmat('examples/Data/300WLP-std_134212_1_0.mat')['pt3d']
    size = 450

    height, width = img.shape[:2]
    '''
    Generate params
    '''
    # n_img, info = fm.generate_3ddfa_params(img, pts, False, shape=(size,size), expand_ratio=1.)

    '''
    Random crop
    '''
    # n_img, n_params = random_crop(n_img, info['roi_box'], info['params'], expand_ratio=1.)
    # re_pts = fm.reconstruct_vertex(n_img, n_params, False)[fm.bfm.kpt_ind]
    # show_pts(n_img, re_pts)

    '''
    Flip vertically
    '''
    # flip(n_img, info['params'])

    '''
    Squeeze face
    '''
    squeeze_face(img, pts, 0.3, 'h')

    '''
    Light
    '''
    # shp, exp, scale, angles, trans = fm._parse_params(info['params'], False)
    # vertices = fm.bfm.reduced_generated_vertices(shp, exp)
    # vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]
    # vertices = mesh.transform.similarity_transform(
    #     vertices, 
    #     scale, 
    #     mesh.transform.angle2matrix([0, 0, 0]), 
    #     [0, 0, 0]) 
    # vertices = fm.reconstruct_vertex(n_img, info['params'], False)
    # vertices[:,1] = height - vertices[:,1] - 1

    # vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]
    # # flip vertices along y-axis.
    
    # light_positions = np.array([[0,-200,300]])
    # light_intensities = np.array([[1,1,1]])
    # light_img, light_vertices = light_test(vertices, light_positions, light_intensities)
    # show_pts(light_img, light_vertices[fm.bfm.kpt_ind], 'BGR')

    '''
    Read params
    '''
    # img = cv2.imread('samples/0168_1.jpg')
    # params = sio.loadmat('samples/0168_1.mat')['params'].reshape(-1,)

    # re_pts = fm.reconstruct_vertex(img, params, False)[fm.bfm.kpt_ind]
    # show_pts(img, re_pts)

    '''
    Rotate params
    '''
    # magic = [75.41140417589962, -79.51944989389769, -72.06898665794476]
    # r_img, r_params = fm.augment_rotate(img, pts, [-60, -70, 0])
    
    # re_pts = fm.reconstruct_vertex(r_img, r_params, False)[fm.bfm.kpt_ind]
    # show_pts(r_img, re_pts)

    # img = cv2.imread('samples/300WLP-std_134212_1_12.jpg')
    # pts = sio.loadmat('samples/300WLP-std_134212_1_12.mat')['pt3d']

    # n_img, info = fm.generate_3ddfa_params(img, pts, False)
    # re_pts = fm.reconstruct_vertex(n_img, info['params'], False)[fm.bfm.kpt_ind]
    # show_pts(n_img, re_pts)


