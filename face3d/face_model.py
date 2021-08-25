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

    def _preprocess_face_landmarks(self, img, pt, expand_ratio=1., shape=(128,128)):
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

        return tddfa_params.reshape(-1,)

    def _parse_params(self, params, de_normalize=True):
        """
        Parse 3DDFA parameters to camera matrix, shape and expression.

        Args:
            :params: 3DDFA parameters.
            :de_normalize: If input params is not coming from train dataset,
                           set this False.
        """
        if de_normalize:
            params = params * self.bfm.params_std_101 + self.bfm.params_mean_101

        camera_matrix = params[:12].reshape(3, -1)
        scale, rotation_matrix, trans = mesh.transform.P2sRt(camera_matrix)
        angles = mesh.transform.matrix2angle(rotation_matrix)

        shp = params[12:72].reshape(-1, 1)
        exp = params[72:].reshape(-1, 1)

        return shp, exp, scale, angles, trans

    def reconstruct_vertex(self, img, params, de_normalize=True):
        """
        Reconstruct a point cloud (~50k vertices) from 3DDFA parameters.

        Args:
            :img: input image.
            :params: 3DDFA parameters.
            :de_normalize: If input params is not coming from train dataset,
                           set this False.

        Return:
            :image_vertices: 3d face point cloud of input image.
        """
        shp, exp, scale, angles, trans = self._parse_params(params, de_normalize)

        h, w, _ = img.shape

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

    def augment_rotate(self, img, pt, angles=[-70,0,0]):
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
        import time
        foo = []
        foo.append(time.time())
        
        params = self.get_3DDFA_params(img, pt)
        
        foo.append(time.time())

        vertices = self.reconstruct_vertex(img, params, de_normalize=False)
        
        foo.append(time.time())

        colors = _get_colors(img, vertices.astype(int))

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
        obj['scale'] = np.float32(1)
        if angles is None:
            angle_choices = [[-i-1,0,0] for i in range(59,70)]
            obj['angles'] = random.choice(angle_choices)
        else:
            obj['angles'] = angles
        obj['trans'] = [0,0,0]

        rotated_img, rotated_vertices = self._transform_test(obj, camera, h, w)

        foo.append(time.time())
        
        params = self.get_3DDFA_params(
            rotated_img, rotated_vertices[self.bfm.kpt_ind][:,:2]
        )

        foo.append(time.time())

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

        for idx in range(len(foo)-1):
            print(foo[idx+1]-foo[idx])

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
    
    def generate_rotated_3d_img(self, img, pt, angles=None, blended=False, shape=(128,128)):
        img, pt = self._preprocess_face_landmarks(img, pt, shape=shape)
        r_img, params = self.augment_rotate(img, pt, angles=angles)

        if blended:
            import time
            t0 = time.time()
            blended_img = utils.blend_smooth_image(
                r_img,
                random.choice(BACKGROUND)
            )
            print(time.time()-t0)
            return blended_img, params
        else:
            return r_img, params

    def generate_3ddfa_params(self, img, pt, preprocess=True, expand_ratio=1.1, shape=(128,128)):
        if preprocess:
            img, pt = self._preprocess_face_landmarks(img, pt, expand_ratio=expand_ratio, shape=shape)
        
        tddfa_params = self.get_3DDFA_params(img, pt)
        roi_box = utils.get_landmarks_wrapbox(pt)

        vertex = self.reconstruct_vertex(
            img, 
            tddfa_params.reshape(-1,),
            de_normalize=False
        )[self.bfm.kpt_ind]
        box_left = np.min(vertex.T[0])
        box_right = np.max(vertex.T[0])
        box_top = np.min(vertex.T[1])
        box_bot = np.max(vertex.T[1])
        print([(box_left+box_right)/2, (box_top+box_bot)/2])
        utils.show_pts(img, vertex)
        import ipdb; ipdb.set_trace(context=10)

        return img, {'params': tddfa_params, 'roi_box': roi_box}
    
    def generate_sample(self, height, width, params, background=None):
        """
        Generate face sample from 3DDFA parameters.
        Color is randomized.

        Args:
            :height: height of generated image.
            :width: width of generatewd image.
            :params: 3DDFA parameters.
            :background: background to fill image.

        Return:
            :img: face image corresponding to the 3DDFA params.
            #TODO: verification need.
        """

        tp = self.bfm.get_tex_para('random')
        vertices = self.reconstruct_vertex(
            np.zeros((height,width,3)), 
            params,
            de_normalize=False
        )

        image = mesh.render.render_colors(
            vertices=vertices, 
            triangles=self.bfm.triangles, 
            colors=self.bfm.generate_colors(tp), 
            h=height,
            w=width,
            BG=background
        )

        return image