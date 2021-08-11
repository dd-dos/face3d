from . import morphable_model
from . import utils
from . import mesh
import numpy as np
import numba
import glob
import os 
import random
BACKGROUND = list(glob.glob('examples/Data/background/*'))

@numba.njit()
def _get_colors(img, img_vertices):
    colors = np.zeros((len(img_vertices),3))

    for idx in range(len(img_vertices)):
        x, y = img_vertices[idx][:2]
        colors[idx] = img[y][x] / 255.

    return colors

class FaceModel:
    def __init__(self, 
                n_shape=60,
                n_exp=29):
        """
        Main class for processing face model.

        Args:
            :n_shape: number of shape parameters.
            :n_exp: number of expression parameters.
        """
        self.bfm = morphable_model.MorphabelModel()
        self.n_shape = n_shape
        self.n_exp = n_exp

    def _get_params(self, img, pt):
        """
        Get 3dmm parameters.

        Args:
            :img: input image.
            :pt: 68 3D landmarks.

        Returns:
            :shp: shape parameters.
            :exp: expression parameters.
            :scale: scale ratio.
            :angles: angles (extrinsic matrix).
            :trans: translation vector.
        """
        h,w,_ = img.shape
        pt.T[1] = h - 1 - pt.T[1]
        pt.T[0] -= w/2
        pt.T[1] -= h/2

        shp, exp, scale, angles, trans = self.bfm.reduced_fit(
            pt, n_sp=self.n_shape, n_ep=self.n_exp, max_iter=3
        )
        return shp, exp, scale, angles, trans

    def _preprocess_face_landmarks(self, img, pt, expand_ratio=1.5, shape=(128,128)):
        """
        Crop then resize image and landmarks to target shape.

        Args:
            :img: input image.
            :pt: 68 3D landmarks.
            :expand_ratio: ratio to expand when crop image.
            :shape: target shape for resizing.

        Returns:
            :img: output image after processing.
            :pt: output landmarks after processing. 
        """
        img, pt = utils.crop_face_landmarks(img, pt, expand_ratio)
        img, pt = utils.resize_face_landmarks(img, pt, shape)

        return img, pt
    
    def get_3DDFA_params(self, img, pt):
        """
        Get 3DDFA parameters.
        Remember to preprocess image and landmarks points if needed.

        Args:
            :img: input image.
            :pt: 68 3D landmarks.

        Returns:
            :tddfa_params: 3DDFA parameters. Training ready.
        """
        shp, exp, scale, angles, trans = self._get_params(img, pt)
        rotation_matrix = mesh.transform.angle2matrix(angles)
        scale_rotation_matrix = scale * rotation_matrix
        camera_matrix = np.concatenate(
            (scale_rotation_matrix, trans.reshape(-1,1)), axis=1
        )
        dense_camera_matrix = camera_matrix.reshape((12,1))
        tddfa_params = np.concatenate((dense_camera_matrix, shp, exp), axis=0)
        
        # extra = {
        #     'shp': shp,
        #     'exp': exp,
        #     'scale': scale,
        #     'angles': angles,
        #     'trans': trans,
        # }

        return tddfa_params

    def reconstruct_vertex(self, img, params):
        """
        Reconstruct a point cloud (~50k vertices) from 3DDFA parameters.

        Args:
            :img: input image.
            :params: 3DDFA parameters.

        Return:
            :image_vertices: 3d face point cloud of input image.
        """
        h, w, _ = img.shape

        camera_matrix = params[:12].reshape(3, -1)
        scale, rotation_matrix, trans = mesh.transform.P2sRt(camera_matrix)
        angles = mesh.transform.matrix2angle(rotation_matrix)

        shp = params[12:72].reshape(-1, 1)
        exp = params[72:].reshape(-1, 1)

        vertices = self.bfm.reduced_generated_vertices(shp, exp)
        transformed_vertices = self.bfm.transform(vertices, scale, angles, trans)
        image_vertices = mesh.transform.to_image(transformed_vertices, h, w)

        return image_vertices

    def _transform_test(self, obj, camera, h = 256, w = 256):
        '''
        Args:
            obj: dict contains obj transform paras
            camera: dict contains camera paras
        '''
        R = mesh.transform.angle2matrix(obj['angles'])
        transformed_vertices = mesh.transform.similarity_transform(
            obj['vertices'], obj['scale'], R, obj['trans']
        )
        
        if camera['proj_type'] == 'orthographic':
            projected_vertices = transformed_vertices
            image_vertices = mesh.transform.to_image(projected_vertices, h, w)
        else:
            ## world space to camera space. (Look at camera.) 
            camera_vertices = mesh.transform.lookat_camera(
                transformed_vertices, camera['eye'], camera['at'], camera['up']
            )
            ## camera space to image space. (Projection) if orth project, omit
            projected_vertices = mesh.transform.perspective_project(
                camera_vertices, camera['fovy'], near = camera['near'], far = camera['far']
            )
            ## to image coords(position in image)
            image_vertices = mesh.transform.to_image(projected_vertices, h, w, True)

        rendering = mesh.render.render_colors(
            image_vertices, self.bfm.triangles, obj['colors'], h, w
        )

        rendering = np.minimum((np.maximum(rendering, 0)), 1)
        return rendering, image_vertices

    def augment_rotate(self, img, pt, angles=[-70,0,0], base_size=180):
        """
        Rotate input image in 3D space.
        Remember to preprocess image and landmarks points if needed.
        
        Args:
            :img: input image.
            :pt: 68 3D landmarks.
            :angles: rotate angles.
            :base_size: human base face size. 1 unit ~ 0.1 cm.
        Returns:
            :rotated_img: rotated image.
            :params: 3DDFA parameters of rotated image.
        """
        params = self.get_3DDFA_params(img, pt)
        vertices = self.reconstruct_vertex(img, params)
        colors = _get_colors(img, vertices.astype(np.uint16))

        h,w,_ = img.shape
        vertices.T[1] = h - 1 - vertices.T[1]
        vertices.T[0] -= w/2
        vertices.T[1] -= h/2

        colors = colors/np.max(colors)
        vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]

        obj = {}
        camera = {}

        camera['proj_type'] = 'orthographic'
        obj['vertices'] = vertices
        obj['colors'] = colors
        obj['scale'] = base_size/(np.max(vertices[:,1]) - np.min(vertices[:,1]))
        if angles is None:
            angle_choices = [[-i-1,0,0] for i in range(59,70)]
            obj['angles'] = random.choice(angle_choices)
        else:
            obj['angles'] = angles
        obj['trans'] = [0, 0, 0]

        rotated_img, rotated_vertices = self._transform_test(obj, camera, h, w)
        params = self.get_3DDFA_params(
            rotated_img, rotated_vertices[self.bfm.kpt_ind][:,:2]
        )
        # new_rotation_matrix = mesh.transform.angle2matrix(list(extra['angles'])+obj['angles'])
        # new_scale_rotation_matrix = extra['scale'] * obj['scale'] * new_rotation_matrix 
        
        # new_trans = obj['scale'] * np.array(extra['trans']) @ mesh.transform.angle2matrix(obj['angles'])
        # new_camera_matrix = np.concatenate(
        #     (new_scale_rotation_matrix, new_trans.reshape(-1,1)), axis=1
        # )
        # new_dense_camera_matrix = new_camera_matrix.reshape((12,1))
        # params[:12] = new_dense_camera_matrix

        # utils.show_pts(rotated_img, rotated_vertices[self.bfm.kpt_ind])
        rotated_img = (rotated_img*255).astype(np.uint8)

        return rotated_img, params

    def foo(self, img, pt):
        # img, pt = self._preprocess_face_landmarks(img, pt, shape=(256,256))
        # params = self.get_3DDFA_params(img, pt)
        # re_pt = self.reconstruct_vertex(img, params)

        # utils.show_pts(img, re_pt[self.bfm.kpt_ind])
        import time
        t0 = time.time()
        img, pt = self._preprocess_face_landmarks(img, pt, shape=(256,256))
        r_img, params = self.augment_rotate(img, pt)
        print(time.time()-t0)

        r_pt = self.reconstruct_vertex(r_img, params)
        utils.show_pts(r_img, r_pt[self.bfm.kpt_ind])
    
    def generate_rotated_3d_img(self, img, pt, angles=None, blended=False):
        img, pt = self._preprocess_face_landmarks(img, pt, shape=(256,256))
        r_img, params = self.augment_rotate(img, pt, angles=angles)

        if blended:
            blended_img = utils.blend_smooth_image(
                r_img,
                random.choice(BACKGROUND)
            )

            return blended_img, params
        else:
            return r_img, params

    def generate_3ddfa_params(self, img, pt):
        img, pt = self._preprocess_face_landmarks(img, pt, expand_ratio=1.1, shape=(256,256))
        tddfa_params = self.get_3DDFA_params(img, pt)
        roi_box = utils.get_landmarks_wrapbox(pt)

        return img, {'params': tddfa_params, 'roi_box': roi_box}
