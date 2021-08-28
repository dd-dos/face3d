import math
import numpy as np

class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0
    
    def push(self, x):
        self.n += 1
    
        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
        
            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0
    
    def standard_deviation(self):
        return np.sqrt(self.variance())

rs = RunningStats()

import tqdm
from pathlib import Path
import scipy.io as sio
file_list = list(Path('ddfa/300VW-3D_cropped_closed_eyes_3ddfa').glob('**/*.mat')) + \
            list(Path('ddfa/300WLP_3ddfa').glob('**/*.mat')) + \
            list(Path('ddfa/300VW-3D_cropped_opened_eyes_3ddfa').glob('**/*.mat')) + \
            list(Path('ddfa/300WLP_3ddfa').glob('**/*.mat')) + \
            list(Path('ddfa/AFLW2000_3ddfa').glob('**/*.mat'))
            
for file_path in tqdm.tqdm(file_list, total=len(file_list)):
    params = sio.loadmat(file_path)['params']
    rs.push(params)

sio.savemat('params_mean_std_12_pose_60_shp_29_exp.mat', {'mean': rs.mean(), 'std': rs.standard_deviation()})