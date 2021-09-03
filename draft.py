from numpy.core.fromnumeric import squeeze
from numpy.random import rand
from face3d.mesh import light
import re
import cv2
import scipy.io as sio
from face3d.utils import *
from face3d.face_model import FaceModel, _get_colors
from face3d import mesh
from utils import close_eyes_68_ver_1, close_eyes_68_ver_2
import numpy as np
import random
fm = FaceModel()

def light_test(vertices, light_positions, light_intensities, h = 256, w = 256, colors=None, light=True):
    if colors is None:
        colors = fm.bfm.generate_colors(fm.bfm.get_tex_para())
        colors = colors/np.max(colors)

    if light == True:   
        lit_colors = mesh.light.add_light(vertices, fm.bfm.triangles, colors, light_positions, light_intensities)
    else:
        lit_colors = colors

    # image_vertices = mesh.transform.to_image(vertices, h, w)
    image_vertices = vertices
    rendering = mesh.render.render_colors(image_vertices, fm.bfm.triangles, lit_colors, h, w)
    rendering = np.minimum((np.maximum(rendering, 0)), 1)
    return rendering, image_vertices

def squeeze_face(img, pts, pad_ratio=None, squeeze_type=None):
    ### Squeeze image and landmarks ###

    height, width = img.shape[:2] 
    size = min(height, width)

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


def random_crop(img, info, expand_ratio=1):
    roi_box = info['roi_box']
    params = info['params']
    camera_matrix = params[:12].reshape(3, -1)
    scale, rotation_matrix, trans = mesh.transform.P2sRt(camera_matrix)

    # Get the box that wrap all landmarks.
    # box_top, box_left, box_bot, box_right = \
    # get_landmarks_wrapbox(landmarks)
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
    crop_size = int(max_length/2 * expand_ratio)

    img_height, img_width, channel = img.shape
    canvas = np.zeros((img_height+2*crop_size, img_width+2*crop_size, channel), dtype=np.uint8)
    canvas[crop_size:img_height+crop_size, crop_size:img_width+crop_size, :] = img
    center[0] += crop_size
    center[1] += crop_size

    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    center_x = int((bbox[2]+bbox[0])/2)
    center_y = int((bbox[3]+bbox[1])/2)

    # Top left bottom right.
    y1 = center_y-crop_size
    x1 = center_x-crop_size
    y2 = center_y+crop_size
    x2 = center_x+crop_size

    re_trans = trans + np.array([crop_size, -crop_size, 0])
    re_params = fm.reconstruct_params(scale, rotation_matrix, re_trans, params[12:])
    re_pts = fm.reconstruct_vertex(img, re_params, False)[fm.bfm.kpt_ind]
    show_pts(canvas, re_pts)

    import ipdb; ipdb.set_trace(context=10)
    cropped_img = canvas[y1:y2, x1:x2]

if __name__=='__main__':
    img = cv2.imread('examples/Data/image00050.jpg')
    pts = sio.loadmat('examples/Data/image00050.mat')['pt3d_68'].T[:,:2]

    # img = cv2.imread('examples/Data/300WLP-std_134212_1_0.jpg')
    # pts = sio.loadmat('examples/Data/300WLP-std_134212_1_0.mat')['pt3d']
    size = 450

    n_img, info = fm.generate_3ddfa_params(img, pts, False, shape=(size,size), expand_ratio=1.)

    random_crop(n_img, info)
    # light_img, light_pts = light_test(re_pts, np.array([[0,-0,-50]]), np.array([[1,1,1]]), size, size, light=False)