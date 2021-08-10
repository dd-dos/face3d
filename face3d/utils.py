import multiprocessing as mp
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numba
import numpy as np
import PIL
import scipy.io as sio
import torchfile
import tqdm
from PIL import Image, ImageFilter, ImageOps


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


def isNumpy(image):
    if isinstance(image, np.ndarray):
        return True
    else:
        return False

def isPIL(image):
    if isinstance(image, PIL.Image.Image):
        return True
    else: 
        return False

def toNumpy(image):
    if isNumpy(image):
        return image
    elif isPIL(image):
        image = np.array(image)
        return image
    else:
        raise TypeError("Only support for np.ndarray or PIL.Image.Image. Got type: {​}​".format(type(image)))


def toPIL(image):
    if isNumpy(image):
        image = Image.fromarray(image)
        return image
    elif isPIL(image):
        return image
    else:
        raise TypeError("Only support for np.ndarray or PIL.Image.Image. Got type: {​}​".format(type(image)))


def get_avg_brightness(image):
    ''' Get average of value of brightness of image
    Params
    ------
    :image: np.ndarray or PIL image
    Returns
    -------
    Average brightness of image
    '''
    image = toNumpy(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return np.mean(v)


def get_avg_saturation(image):
    ''' Get average of value of saturation of image
    Params
    ------
    :image: np.ndarray or PIL image
    Returns
    -------
    Average saturation of image
    '''
    image = toNumpy(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return np.mean(s)


def change_brightness(image, value=1.0):
    is_numpy = isNumpy(image)
    image = toNumpy(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    v = value * v
    v[v > 255] = 255
    v = np.asarray(v, dtype=np.uint8)
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    image = toNumpy(image) if is_numpy else toPIL(image)
    return image


def change_saturation(image, value):
    is_numpy = isNumpy(image)
    image = toNumpy(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    s = value * s
    s[s > 255] = 255
    s = np.asarray(s, dtype=np.uint8)
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    image = toNumpy(image) if is_numpy else toPIL(image)
    return image


def matching_brightness(src_image, dst_image):
    ''' Matching brightness of src_image to brightness of dst_image
    Params
    ------
    src_image: np.ndarray or PIL image
    dst_image: np.ndarray or PIL image
    Returns
    -------
    result image
    '''    
    is_numpy = isNumpy(src_image)
    raw_src = toNumpy(src_image)
    raw_dst = toNumpy(dst_image)
    rgba = False
    if raw_src.shape[-1] == 4:
        alpha_channel = raw_src[:,:,3].copy()
        raw_src = raw_src[:,:, :3]
        rgba = True
    raw_image = raw_src[:, :, :3].copy()
    src_brightness = get_avg_brightness(raw_src)
    dst_brightness = get_avg_brightness(raw_dst)
    delta_brightness = 1 + (dst_brightness - src_brightness)/255
    src_saturation = get_avg_saturation(raw_src)
    dst_saturation = get_avg_saturation(raw_dst)
    delta_saturation = 1 + (dst_saturation - src_saturation)/255
    raw_image = change_brightness(raw_image, delta_brightness)
    raw_image = change_saturation(raw_image, delta_saturation)
    if rgba:
        alpha_channel = np.expand_dims(alpha_channel, axis=2)
        raw_image = np.concatenate((raw_image, alpha_channel), axis=-1)
    raw_image = toNumpy(raw_image) if is_numpy else toPIL(raw_image)
    return raw_image


def create_transparent_image(image, threshold, mode='equal', get_full_object=False):
    ''' Create transparent image
    Params
    ------
    :image: np.ndarray or PIL image
    :threshold: threshold to transparent
    :model: 
        'equal' - all pixels have value equal to threshold will become transparent
        'greater' - all pixels have value greater than threshold will become transparent
        'lower' - all pixels have value lower than threshold will become transparent
    :get_full_object: Keep 1 main object and all pixels in object
    Returns
    -------
    transparent image
    '''
    is_numpy = isNumpy(image)
    if isinstance(threshold, int):
        threshold = (threshold,threshold, threshold)
    elif not (isinstance(threshold, tuple) or isinstance(threshold,list)):
        raise TypeError("Type of threshold must be int or list or tuple. Got{}".format(threshold))
    image_rgba = toNumpy_RGBA(image)
    image_rgb = image_rgba[:,:,:3].copy()
    # Transparent mask
    if mode == "equal":
        # transparent_area = np.any(image_rgb == threshold, axis=-1)
        mask_1 = image_rgb[:,:,0] == threshold[0]
        mask_2 = image_rgb[:,:,1] == threshold[1]
        mask_3 = image_rgb[:,:,2] == threshold[2]
    elif mode == "greater":
        # transparent_area = np.any(image_rgb >= threshold, axis=-1)
        mask_1 = image_rgb[:,:,0] >= threshold[0]
        mask_2 = image_rgb[:,:,1] >= threshold[1]
        mask_3 = image_rgb[:,:,2] >= threshold[2]
        
    elif mode == "lower":
        transparent_area = np.any(image_rgb <= threshold, axis=-1)
        mask_1 = image_rgb[:,:,0] <= threshold[0]
        mask_2 = image_rgb[:,:,1] <= threshold[1]
        mask_3 = image_rgb[:,:,2] <= threshold[2]
    else:
        raise TypeError("Valid mode in ['equal', 'greater', 'lower']. Got {}".format(mode))
    transparent_area = mask_1 * mask_2 * mask_3
    if get_full_object:
        # Get the main object and all pixels in that object
        invert_mask = np.invert(transparent_area)
        transparent_area_int = invert_mask.astype(np.uint8) * 255
        mask_rgb = cv2.cvtColor(transparent_area_int, cv2.COLOR_GRAY2BGR)
        _, bin_mask = cv2.threshold(transparent_area_int, 127, 255, 0)
        cnts, _ = cv2.findContours(bin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Get biggest contour
        large_cnt = max(cnts, key = cv2.contourArea)
        transparent_area = np.zeros(transparent_area.shape)
        cv2.fillPoly(transparent_area, pts=[large_cnt], color=(255,255,255))
        transparent_area = np.invert(np.bool_(transparent_area))
        
    image_rgba[transparent_area, -1] = 0 
    image_rgba = toNumpy(image_rgba) if is_numpy else toPIL(image_rgba)
    return image_rgba
     

def blend_smooth_image(image, background, xy=(0,0), wh=None, iterations=None, smooth_mode=True, brightness_matching=True):
    ''' Blend image to background with smoothly effect
    Params
    ------
    :image: np.ndarray or PIL image
    :background: np.ndarray or PIL image
    :xy: top-left coordinate of image on background
    Returns
    -------
    Blended image: np.ndarray or PIL image
    '''
    # NOTE: Need to update blend image with rgba image.
    # Drop transparent area and keep the rest area(polygon).
    if isinstance(image, str):
        image = cv2.imread(str)
    
    if isinstance(background, str):
        background = cv2.imread(background)

    is_numpy = isNumpy(image)
    background = toPIL(background)
    image = toPIL(image)

    w, h = image.size
    background = background.resize((w,h))

    image = create_transparent_image(image, threshold=50, mode='lower')

    left, top = xy
    if brightness_matching:
        image = matching_brightness(src_image=image, dst_image=background)
    if iterations is None:
        iterations = random.randint(2, 4)
    if wh is not None:
        wid, hei = wh
        image = image.resize((wid, hei), Image.BICUBIC)
    else:
        wid, hei = image.size
    raw_background = background.copy()
    # Create  smoothly mask 
    if image.mode == 'RGBA':
        # Let alpha channel of image is mask
        alpha = image.split()[-1]
        mask_alpha = alpha.copy()
        mask_alpha = mask_alpha.point(lambda p: p > 10 and 255)
        mask_alpha = mask_alpha.convert("RGB")
        mask = Image.new("RGB", (background.size), (0, 0, 0))
        mask.paste(mask_alpha, (left, top))
    else:
        mask = Image.new("RGB", (background.size), (0, 0, 0))
        mask.paste((255,255,255), (left, top, left+wid, top+hei))
    for _ in range(iterations):
        mask = mask.filter(ImageFilter.BLUR)
    mask_np = np.array(mask) /255
    # Insert image to background
    if image.mode == 'RGBA':
        background.paste(image, (left, top), image)
    else:
        background.paste(image, (left, top))
    # If do not use smooth_mode, return raw blend image
    if not smooth_mode:
        background = toNumpy(background) if is_numpy else toPIL(background)
        return background
    # Apply smooth effect
    sharp_image_np = np.array(background)
    background_np = np.array(raw_background)
    # Blend 2 image with smooth mask
    smooth_image = (sharp_image_np * mask_np + background_np*(1-mask_np)).astype(np.uint8)
    smooth_image_blur = cv2.blur(smooth_image, (3,3))
    alpha = 0.5
    smooth_image = cv2.addWeighted(smooth_image, alpha, smooth_image_blur, 0.5, 0)
    smooth_image = toNumpy(smooth_image) if is_numpy else toPIL(smooth_image)
    return smooth_image


def toPIL_RGBA(image):
    image = toPIL(image)
    if image.mode == 'RGBA':
        return image
    else:
        image_rgba = image.convert(image, "RGBA")
        return image_rgba


def toNumpy_RGBA(image, alpha_value=255):
    image = toNumpy(image)
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            return image
        elif image.shape[2] == 3:
            channel_1, channel_2, channel_3 = cv2.split(image)
            alpha_channel = np.ones(channel_1.shape, dtype=channel_1.dtype) * alpha_value #creating a dummy alpha channel image.
            image_RGBA = cv2.merge((channel_1, channel_2, channel_3, alpha_channel))
            return image_RGBA
    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        return image

def isgray(img):
    if len(img.shape) < 3: return True
    if img.shape[2]  == 1: return True
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    return False


def dis(pt1, pt2):
    return np.linalg.norm(pt1-pt2)


def check_frontal_face(pts, threshold=3):
    if np.abs(dis(pts[0], pts[27]) - dis(pts[27], pts[16])) <= threshold and \
        dis(pts[0], pts[16]) > dis(pts[0], pts[27]) and \
        dis(pts[0], pts[16]) > dis(pts[27], pts[16]):
        if pts[0][1] < pts[8][1] and pts[16][1] < pts[8][1]:
            return True
        else:
            return False
    else:
        return False 


def task(img_path):
    img_path = str(img_path)
    img = cv2.imread(img_path)
    if isgray(img):
        return

    try:
        pts_path = img_path.replace('jpg', 't7')
        if not os.path.isfile(pts_path):
            pts_path = img_path.replace('jpg', 'mat')
            pts = sio.loadmat(pts_path)['pt3d_68'][:2].T
        else:
            pts = torchfile.load(pts_path)
    except Exception as e:
        print(e)
        return

    if check_frontal_face(pts):
        for pt in pts:
            pt = tuple(pt.astype(np.uint16))
            cv2.circle(img, pt, 2, (0,255,0), -1, 10)

        token = img_path.split('/')
        folder_id = token[-2]
        file_id = token[-1]

        os.makedirs(f'frontal_faces/{PARENT}/{folder_id}', exist_ok=True)
        out_path = f'frontal_faces/{PARENT}/{folder_id}/{file_id}'
        cv2.imwrite(out_path, img)
        


if __name__=='__main__':
    global PARENT
    os.makedirs(f'frontal_faces/', exist_ok=True)

    # print('Process AFLW2000')
    # img_list = list(Path('AFLW2000').glob('**/*.jpg'))
    # PARENT = 'AFLW2000'
    # os.makedirs(f'frontal_faces/{PARENT}', exist_ok=True)
    # with mp.Pool(mp.cpu_count()) as p:
    #     r = list(tqdm.tqdm(p.imap(task, img_list), total=len(img_list)))

    # print('Process AFLW2000-3D-Reannotated')
    # img_list = list(Path('AFLW2000-3D-Reannotated').glob('**/*.jpg'))
    # PARENT = 'AFLW2000-3D-Reannotated'
    # os.makedirs(f'frontal_faces/{PARENT}', exist_ok=True)
    # with mp.Pool(mp.cpu_count()) as p:
    #     r = list(tqdm.tqdm(p.imap(task, img_list), total=len(img_list)))

    print('Process 300VW-3D')
    img_list = list(Path('300VW-3D').glob('**/*.jpg'))
    PARENT = '300VW-3D'
    os.makedirs(f'frontal_faces/{PARENT}', exist_ok=True)
    with mp.Pool(mp.cpu_count()) as p:
        r = list(tqdm.tqdm(p.imap(task, img_list), total=len(img_list)))

    # path_img = 'black_background_face_path'         
    # image = Image.open(path_img)
    # image = create_transparent_image(image, threshold=50, mode='lower')
    # path_bg = 'path_back_ground'
    # background = Image.open(path_bg)
    # coord_paste = (x, y)
    # blend_image = blend_smooth_image(image, background, xy=(x, y), smooth_mode=True, iterations=5)
    
    
  
  
