import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from retinex import automatedMSRCR

if __name__ == '__main__':
    root_dir = 'D:\\Datasets\\mini-casia\\train'
    images = ['fake/train_1_3_f1.jpg', 'fake/train_1_3_f2.jpg',
              'fake/train_1_4_f1.jpg', 'fake/train_1_4_f2.jpg',
              # 'fake/train_1_5_f1.jpg', 'fake/train_1_5_f2.jpg',
              # 'fake/train_1_6_f1.jpg', 'fake/train_1_6_f2.jpg',
              'real/train_1_1_f1.jpg', 'real/train_1_1_f2.jpg',
              'real/train_1_2_f1.jpg', 'real/train_1_2_f2.jpg',
              # 'real/train_1_HR_f3.jpg', 'real/train_1_HR_f4.jpg',
              # 'real/train_1_HR_f4.jpg', 'real/train_1_HR_f5.jpg',
              ]

    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 4

    for index, img in enumerate(images):
        img = cv2.imread(os.path.join(root_dir, img))
        img = cv2.resize(img, (224, 224))

        fig.add_subplot(4, 4, 2 * index + 1)
        tmp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(tmp)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, -1)
        new_img = automatedMSRCR(img, [2, 4, 8])
        # new_img = multiScaleRetinex(img, [5, 10, 15])
        # print(new_img.shape)
        # new_img = cv2.cvtColor(new_img[:,:,0], cv2.COLOR_GRAY2RGB)
        fig.add_subplot(4, 4, 2 * index + 2)
        plt.imshow(new_img[:, :, 0], cmap='gray')
    plt.show()

# img = multiScaleRetinex(img, [10, 20, 30])
# img = automatedMSRCR(img, [10,20,30])
# print(img)
# print(img.shape)
# cv2.imshow("img", img)
# cv2.waitKey(0)
