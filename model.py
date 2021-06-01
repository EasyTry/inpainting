import numpy as np
import torch


from DeepFlowInpainting.tools.frame_inpaint import DeepFillv1

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

class Inpainter:
    def __init__(self, height):
        checkpoint_dir = './pretrained_weights/'
        self.model_restore = checkpoint_dir + 'imagenet_deepfill.pth'
        self.model = DeepFillv1(pretrained_model=self.model_restore,
                          image_shape=(height, height),
                          res_shape=None)

    def __call__(self, img, mask):
        h, w = img.shape[:2]
        if h < w:
            print('Unexpected image shape')
        padded_img = np.pad(img, ((0, 0), (0, h - w), (0, 0)), mode='constant')
        if len(mask.shape) == 3:
            padded_mask = np.pad(mask, ((0, 0), (0, h - w), (0, 0)), mode='constant')
        elif len(mask.shape) == 2:
            padded_mask = np.pad(mask, ((0, 0), (0, h - w)), mode='constant')
        with torch.no_grad():
            padded_img_res = self.model.forward(padded_img, padded_mask)
        img_res = np.array(padded_img_res[:h, :w], dtype=int)
        return img_res
