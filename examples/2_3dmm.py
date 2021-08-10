''' 3d morphable model example
3dmm parameters --> mesh 
fitting: 2d image + 3dmm -> 3d face
'''
import os, sys
import subprocess
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

sys.path.append('..')
from face3d import mesh
from face3d.morphable_model import MorphabelModel

import cv2
from PIL import Image

def show_ndarray_img(img):
    if np.mean(img) <= 1:
        img = (img*255).astype(np.uint8)

    _img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    Image.fromarray(_img).show()


def show_vertices(vertices: np.ndarray, v_type='3D'):
    if v_type=='3D':
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        _vertices = vertices.transpose(1, 0)
        ax.scatter(_vertices[0],
                   _vertices[1],
                   _vertices[2],
                   marker=".")

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

# --------------------- Forward: parameters(shape, expression, pose) --> 3D obj --> 2D image  ---------------
# --- 1. load model
bfm = MorphabelModel('Data/BFM/Out/BFM.mat')
import ipdb; ipdb.set_trace(context=10)
print('init bfm model success')

# --- 2. generate face mesh: vertices(represent shape) & colors(represent texture)
sp = bfm.get_shape_para('random')
ep = bfm.get_exp_para('random')
tp = bfm.get_tex_para('random')

# info = sio.loadmat('/home/pdd/Desktop/Workspace/3DDFA-1/face3d/examples/Data/IBUG_image_008_1_0.mat')
# sp = info['Shape_Para']
# ep = info['Exp_Para']
# tp = info['Tex_Para']

vertices = bfm.generate_vertices(sp, ep)
colors = bfm.generate_colors(tp)
colors = np.minimum(np.maximum(colors, 0), 1)
import ipdb; ipdb.set_trace(context=10)
# --- 3. transform vertices to proper position
s = 8e-04
angles = [10, 30, 20]
t = [0, 0, 0]

# pose_param = info['Pose_Para'][0]
# s = pose_param[-1]
# angles = pose_param[3:6]
# t = pose_param[0:3]

transformed_vertices = bfm.transform(vertices, s, angles, t)
projected_vertices = transformed_vertices.copy() # using stantard camera & orth projection

# --- 4. render(3d obj --> 2d image)
# set prop of rendering
h = w = 256; c = 3
image_vertices = mesh.transform.to_image(projected_vertices, h, w)
image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)

# -------------------- Back:  2D image points and corresponding 3D vertex indices-->  parameters(pose, shape, expression) ------
## only use 68 key points to fit
x = projected_vertices[bfm.kpt_ind, :2] # 2d keypoint, which can be detected from image
import ipdb; ipdb.set_trace(context=10)

# x[37] = x[41]
# x[38] = x[40]
# x[43] = x[47]
# x[44] = x[46]
# x = x / np.max(x) * 256
# x -= np.mean(x)
X_ind = bfm.kpt_ind # index of keypoints in 3DMM. fixed.

# fit
fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = \
    bfm.fit(x, X_ind, max_iter = 3)

# verify fitted parameters
fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
transformed_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)

image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)


# ------------- print & show 
print('pose, groudtruth: \n', s, angles[0], angles[1], angles[2], t[0], t[1])
print('pose, fitted: \n', fitted_s, fitted_angles[0], fitted_angles[1], fitted_angles[2], fitted_t[0], fitted_t[1])

save_folder = 'results/3dmm'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

io.imsave('{}/generated.jpg'.format(save_folder), image)
io.imsave('{}/fitted.jpg'.format(save_folder), fitted_image)


### ----------------- visualize fitting process
# fit
fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(x, X_ind, max_iter = 3, isShow = True)

# verify fitted parameters
for i in range(fitted_sp.shape[0]):
	fitted_vertices = bfm.generate_vertices(fitted_sp[i], fitted_ep[i])
	transformed_vertices = bfm.transform(fitted_vertices, fitted_s[i], fitted_angles[i], fitted_t[i])

	image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
	fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)
	io.imsave('{}/show_{:0>2d}.jpg'.format(save_folder, i), fitted_image)

options = '-delay 20 -loop 0 -layers optimize' # gif. need ImageMagick.
subprocess.call('convert {} {}/show_*.jpg {}'.format(options, save_folder, save_folder + '/3dmm.gif'), shell=True)
subprocess.call('rm {}/show_*.jpg'.format(save_folder), shell=True)
