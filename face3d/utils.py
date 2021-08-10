import numpy as np
import matplotlib.pyplot as plt

import cv2
from PIL import Image
import numba

def show_ndarray_img(img):
    if np.mean(img) <= 1:
        img = (img*255).astype(np.uint8)

    _img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    Image.fromarray(_img).show()


def show_vertices(vertices: np.ndarray, type='3D'):
    if type=='3D':
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        _vertices = vertices.transpose(1, 0)
        ax.scatter(_vertices[0],
                   _vertices[1],
                   _vertices[2],
                   marker=".")

        plt.show() 
        plt.close()
    elif type=='2D':
        # ax, fig = plt.figure()

        # _vertices = vertices.transpose(1, 0)
        # ax.scatter(_vertices[0],
        #            _vertices[1],
        #            marker=".")
        _vertices = vertices.transpose(1, 0)
        plt.scatter(_vertices[0],
                    _vertices[1],
                    marker='.')
        plt.show() 
        plt.close()        
    else:
        return


def show_pts(img, pts):
    if np.mean(img) <= 1:
        img = (img*255).astype(np.uint8)

    img = np.ascontiguousarray(img, dtype=np.uint8)
    _img = img.copy()

    try:
        for i in range(pts.shape[0]):
            _pts = pts[i].astype(int)
            _img = cv2.circle(_img, (_pts[0], _pts[1]),3,(0,255,0), -1, 8)
    except Exception as e:
        print(e)
        import ipdb; ipdb.set_trace(context=10)
    
    _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    Image.fromarray(_img).show()


@numba.njit()
def crop_face_landmarks(img, landmarks, expand_ratio=1.0):
    """
    Pad and crop to retain landmarks when rotating.

    Params:
    :img: image to pad.
    :landmarks: 68 landmarks points.
    """
    # Get the box that wrap all landmarks.
    # box_top, box_left, box_bot, box_right = \
    # get_landmarks_wrapbox(landmarks)
    box_left = int(np.ceil(np.min(landmarks.T[0])))
    box_right = int(np.ceil(np.max(landmarks.T[0])))
    box_top = int(np.ceil(np.min(landmarks.T[1])))
    box_bot = int(np.ceil(np.max(landmarks.T[1])))

    box_height = box_bot-box_top
    box_width = box_right-box_left
    
    # Crop image to get the largest square region that satisfied:
    # 1. Contains all landmarks
    # 2. Center of the landmarks box is the center of the region.
    center = [int(np.ceil((box_left+box_right)/2)), int(np.ceil((box_top+box_bot)/2))]
    
    # Get the diameter of largest region 
    # that a landmark can reach when rotating.
    max_length = int(np.ceil(np.sqrt(np.power(box_height,2)+np.power(box_width,2))))

    # Crop a bit larger.
    crop_size = int(max_length/2 * expand_ratio)

    img_height, img_width, channel = img.shape
    canvas = np.zeros((img_height+2*crop_size, img_width+2*crop_size, channel), dtype=np.uint8)
    canvas[crop_size:img_height+crop_size, crop_size:img_width+crop_size, :] = img

    # Adjust center coord.
    center[0] += crop_size
    center[1] += crop_size

    # Top left bottom right.
    y1 = center[1]-int(crop_size)
    x1 = center[0]-int(crop_size)
    y2 = center[1]+int(crop_size)
    x2 = center[0]+int(crop_size)

    # Crop image.
    img = canvas[y1:y2, x1:x2]
    
    # Adjust landmarks and center
    landmarks.T[0] = landmarks.T[0] - x1 + crop_size
    landmarks.T[1] = landmarks.T[1] - y1 + crop_size

    return img, landmarks


def resize_face_landmarks(img, landmarks, shape=(256,256)):
    height, width, _ = img.shape

    width_ratio = shape[0] / width
    height_ratio = shape[1] / height

    img = cv2.resize(img, shape)

    landmarks.T[0] = landmarks.T[0]*width_ratio
    landmarks.T[1] = landmarks.T[1]*height_ratio

    return img, landmarks 
