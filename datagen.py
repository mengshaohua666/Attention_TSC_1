import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from retinex import automatedMSRCR


class CasiaDataset(Dataset):

    def __init__(self, data_root, mode='train', transform=None):
        """Initialization"""
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.data = []
        self._prepare_data()

    def _prepare_data(self):
        mode_path = os.path.join(self.data_root, self.mode)
        for image_fold in os.listdir(mode_path):
            for image in os.listdir(os.path.join(mode_path, image_fold)):
                if image_fold == 'fake':
                    self.data.append((os.path.join(mode_path, image_fold, image), 1))
                else:
                    self.data.append((os.path.join(mode_path, image_fold, image), 0))

    def __getitem__(self, index):
        """Generate one batch of data"""
        image_path, label = self.data[index]
        # img = self.load_img(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        tmp = self._tensor_to_gray(image)

        image_msr = automatedMSRCR(tmp, [10, 20, 30])

        return {'rgb': image, 'msr': image_msr, 'label': label}

    def _tensor_to_gray(self, image):
        inp = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        inp = inp.astype(np.float32)
        gray_image = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
        gray_image = np.expand_dims(gray_image, -1)
        return gray_image

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.data)
