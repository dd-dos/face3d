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
fm = FaceModel(n_shape=40, n_exp=20)
import numba
from scipy import ndimage

# @numba.njit()
def _get_colors_depth(img, img_vertices):
    colors = np.zeros((len(img_vertices),3))

    depths = img_vertices.T[-1]
    depths = depths/np.max(depths)
    for idx in range(len(img_vertices)):
        value = depths[idx]
        colors[idx] = np.array([value, value, value])

    return colors

def light_test(vertices, light_positions, light_intensities, h = 256, w = 256, colors=None, light=True, depth=False):
    if colors is None:
        if depth:
            colors = _get_colors_depth()
        else:
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

# @numba.njit()
def random_crop_substep(img, roi_box, params, expand_ratio=None, target_size=None, radius=None):
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

    if radius is None:
        radius = max(box_height, box_width) / 2

    max_length = 2*np.sqrt(2)*radius

    # Crop a bit larger.
    if expand_ratio is None:
        expand_ratio = random.uniform(0.8, 1.1)
    else:
        expand_ratio = expand_ratio

    crop_size = int(max_length/2 * expand_ratio)

    img_height, img_width, channel = img.shape
    canvas = np.zeros((img_height+2*crop_size, img_width+2*crop_size, channel), dtype=np.uint8)
    canvas[crop_size:img_height+crop_size, crop_size:img_width+crop_size, :] = img

    # shift_value = int(max_length/2 * expand_ratio - radius)
    '''
    0.125 is purely selected from visualization.
    '''
    # shift_value_x = int(box_width * 0.125 + shift_value)
    # shift_value_y = int(box_height * 0.125 + shift_value)
    # shift_value_x = shift_value
    # shift_value_y = shift_value

    # shift_x = random.randrange(-shift_value_x, shift_value_x)
    # shift_y = random.randrange(-shift_value_y, shift_value_y)

    # shift_x = shift_value_x
    # shift_y = shift_value_y

    center_x = int(center[0] + crop_size)
    center_y = int(center[1] + crop_size)

    # Top left bottom right.
    y1 = center_y-crop_size
    x1 = center_x-crop_size
    y2 = center_y+crop_size
    x2 = center_x+crop_size

    n_box_left = box_left + crop_size - x1
    n_box_right = box_right + crop_size - x1
    n_box_top = box_top + crop_size - y1
    n_box_bot = box_bot + crop_size - y1
    n_roi_box = [n_box_left, n_box_top, n_box_right, n_box_bot]

    cropped_img = canvas[y1:y2, x1:x2]

    flip_matrix = np.array([[1,0,0],[0,-1,0],[0,0,1]], dtype=np.float64)
    flip_offset = np.array([0, img_height, 0], dtype=np.float64)
    norm_trans = np.array([img_width/2, img_height/2, 0], dtype=np.float64)

    cropped_trans = (flip_offset + np.array([-x1, -y1, 0])).reshape(3,) @ flip_matrix.T + norm_trans + trans

    if target_size is None:
        resized_scale = scale
        resized_trans = cropped_trans
    else:
        resized_scale = scale / (2*crop_size) * target_size
        resized_trans = cropped_trans / (2*crop_size) * target_size

    re_scaled_rot_matrix = resized_scale * rotation_matrix
    re_camera_matrix = np.concatenate((re_scaled_rot_matrix, resized_trans.reshape(-1,1)), axis=1)
    re_params = np.concatenate((re_camera_matrix.reshape(12,1), params[12:].reshape(-1,1)), axis=0)

    return cropped_img, re_params, n_roi_box

def random_crop(img, roi_box, params, expand_ratio=None, target_size=None, radius=None):
    '''
    Random crop and resize image to target size.
    '''
    cropped_img, re_params, n_roi_box = random_crop_substep(img, roi_box, params, expand_ratio, target_size, radius)

    if target_size is None:
        re_img = cropped_img
        re_roi_box = np.array(n_roi_box)
    else:
        re_img = cv2.resize(cropped_img, (target_size, target_size))
        re_roi_box = np.array(n_roi_box) / cropped_img.shape[0] * target_size
    # re_pts = fm.reconstruct_vertex(re_img, re_params)[fm.bfm.kpt_ind][:,:2]
    # draw_pts(re_img, re_pts)
    # import ipdb; ipdb.set_trace(context=10)

    return re_img, re_params, re_roi_box
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

def pointcloud_to_depth_map(pointcloud: np.ndarray, theta_res=150, phi_res=32, max_depth=50, phi_min_degrees=60,
                            phi_max_degrees=100) -> np.ndarray:
    """
        All params are set so they match default carla lidar settings
    """
    assert pointcloud.shape[1] == 3, 'Must have (N, 3) shape'
    assert len(pointcloud.shape) == 2, 'Must have (N, 3) shape'

    xs = pointcloud[:, 0]
    ys = pointcloud[:, 1]
    zs = pointcloud[:, 2]

    rs = np.sqrt(np.square(xs) + np.square(ys) + np.square(zs))

    phi_min = np.deg2rad(phi_min_degrees)
    phi_max = np.deg2rad(phi_max_degrees)
    phi_range = phi_max - phi_min
    phis = np.arccos(zs / rs)

    THETA_MIN = -np.pi
    THETA_MAX = np.pi
    THETA_RANGE = THETA_MAX - THETA_MIN
    thetas = np.arctan2(xs, ys)

    phi_indices = ((phis - phi_min) / phi_range) * (phi_res - 1)
    phi_indices = np.rint(phi_indices).astype(np.int16)

    theta_indices = ((thetas - THETA_MIN) / THETA_RANGE) * theta_res
    theta_indices = np.rint(theta_indices).astype(np.int16)
    theta_indices[theta_indices == theta_res] = 0

    normalized_r = rs / max_depth

    canvas = np.ones(shape=(theta_res, phi_res), dtype=np.float32)
    # We might need to filter out out-of-bound phi values, if min-max degrees doesnt match lidar settings
    canvas[theta_indices, phi_indices] = normalized_r

    return canvas

if __name__=='__main__':
    img = cv2.imread('examples/Data/image00050.jpg')
    pts = sio.loadmat('examples/Data/image00050.mat')['pt3d_68'].T[:,:2]

    # img = cv2.imread('examples/Data/300WLP-std_134212_1_0.jpg')
    # pts = sio.loadmat('examples/Data/300WLP-std_134212_1_0.mat')['pt3d']
    size = 450

    height, width = img.shape[:2]
    '''
    Close eyes
    '''
    # pts[37] = pts[41]
    # pts[38] = pts[40]
    # pts[43] = pts[47]
    # pts[44] = pts[46]
    # n_img, info = fm.generate_3ddfa_params(img, pts, False, shape=(size,size), expand_ratio=1.)
    # re_pts = fm.reconstruct_vertex(n_img, info['params'], False)[fm.bfm.kpt_ind]
    # show_vertices(re_pts)

    '''
    Generate params
    '''
    n_img, info = fm.generate_3ddfa_params(img, pts, False, shape=(size,size), expand_ratio=1.)

    '''
    Random crop
    '''
    # for _ in range(10):
    #     t0 = time.time()
    #     n_img, n_params,_ = random_crop(n_img, info['roi_box'], info['params'], expand_ratio=10, radius=300, target_size=128)
    #     print(time.time()-t0)
    # re_pts = fm.reconstruct_vertex(n_img, n_params, False)[fm.bfm.kpt_ind]
    # show_pts(n_img, re_pts)

    '''
    Flip vertically
    '''
    # flip(n_img, info['params'])

    '''
    Squeeze face
    '''
    # squeeze_face(img, pts, 0.3, 'h')

    '''
    Light
    '''
    shp, exp, scale, angles, trans = fm._parse_params(info['params'], False)
    vertices = fm.bfm.reduced_generated_vertices(shp, exp)
    vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]
    vertices = mesh.transform.similarity_transform(
        vertices, 
        scale, 
        mesh.transform.angle2matrix([0, 0, 0]), 
        [0, 0, 0]) 
    vertices = fm.reconstruct_vertex(n_img, info['params'], False).astype(int)
    colors = _get_colors_depth(n_img, vertices)

    vertices[:,1] = height - vertices[:,1] - 1

    vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]
    # flip vertices along y-axis.
    
    light_positions = np.array([[0,-200,300]])
    light_intensities = np.array([[1,1,1]])
    light_img, light_vertices = light_test(vertices, light_positions, light_intensities, light=False, colors=colors, h=450, w=450)
    show_ndarray_img(light_img)

    '''
    Read params
    '''
    # img = cv2.imread('samples/300WLP-std_134212_1_0.jpg')
    # params = sio.loadmat('samples/300WLP-std_134212_1_0.mat')['params'].reshape(-1,)

    # re_pts = fm.reconstruct_vertex(img, params, False)[fm.bfm.kpt_ind]
    # show_pts(img, re_pts)

    '''
    Rotate params
    '''
    # magic = [75.41140417589962, -79.51944989389769, -72.06898665794476]
    # r_img, r_params = fm.augment_rotate(img, pts, [40, -20, -30])
    
    # re_pts = fm.reconstruct_vertex(r_img, r_params, False)[fm.bfm.kpt_ind]
    # show_pts(r_img, re_pts)

    # img = cv2.imread('samples/300WLP-std_134212_1_12.jpg')
    # pts = sio.loadmat('samples/300WLP-std_134212_1_12.mat')['pt3d']

    # n_img, info = fm.generate_3ddfa_params(img, pts, False)
    # re_pts = fm.reconstruct_vertex(n_img, info['params'], False)[fm.bfm.kpt_ind]
    # show_pts(n_img, re_pts)

    '''
    Clip params
    '''
    # img = cv2.imread('samples/0560_0.jpg')
    # params = sio.loadmat('samples/0560_0.mat')['params']
    # shp, exp, scale, angles, trans = fm._parse_params(params.reshape(-1,), False)
    # clipped_shp = shp[:40]
    # clipped_exp = exp[:20]
    # clipped_shp_exp = np.concatenate((clipped_shp, clipped_exp), axis=0)
    # import ipdb; ipdb.set_trace(context=10)
    # clipped_params = fm.reconstruct_params(scale, mesh.transform.angle2matrix(angles),trans, clipped_shp_exp)
    # re_pts = fm.reconstruct_vertex(img, clipped_params, False)[fm.bfm.kpt_ind]
    # show_vertices(re_pts)

