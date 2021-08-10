from numpy.lib.arraysetops import isin
import torch
import torchvision
import os
import cv2
import numpy as np
import os
CWD = os.path.dirname(os.path.abspath(__file__))

class FaceDetector:
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == torch.device('cuda'):
            model_path = os.path.join(CWD, 'weights/scripted_model_gpu_19042021.pt')
        else:
            model_path = os.path.join(CWD, 'weights/scripted_model_cpu_19042021.pt')
    
        self.model = torch.jit.load(model_path)

    @torch.no_grad()
    def predict(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)

        img = torch.tensor(img)
        
        preds = self.model(img)[0]
        detected_faces = [det for det in preds if det[-1] >= 0.9]

        return detected_faces

    @torch.no_grad()
    def predict_batch(self, img_list):
        for idx in range(len(img_list)):
            if isinstance(img_list[idx], str):
                temp = cv2.imread(img_list[idx])
                img_list[idx] = torch.tensor(temp)
            else:
                img_list[idx] = torch.tensor(img_list[idx])

        return self.model.forward_batch(img_list)

if __name__=='__main__':
    img = cv2.imread('/home/pdd/Desktop/Workspace/3DDFA-1/face3d/examples/Data/300WLP-std_134212_1_0.jpg')
    model = FaceDetector()
    preds = model.predict_batch([img, img])
    for pred in preds:
        if pred[0][-1] >= 0.9:
            pred = np.array(pred).astype(np.uint16)
            cv2.rectangle(img, pred[:2], pred[2:-1], color=(0,255,0), thickness=1)
    
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
