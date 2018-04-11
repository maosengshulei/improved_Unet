import cv2
import os
from skimage import img_as_ubyte

def pad(dir_Path,new_root):
    if not os.path.exists(new_root):
        os.mkdir(new_root)
    for root,dirs,files in os.walk(dir_Path):
        for fileobj in files:
            img=cv2.imread(os.path.join(root,fileobj))
            height, width, _ = img.shape
            if height % 32 == 0:
                y_min_pad = 0
                y_max_pad = 0
            else:
                y_pad = 32 - height % 32
                y_min_pad = int(y_pad / 2)
                y_max_pad = y_pad - y_min_pad

            if width % 32 == 0:
                x_min_pad = 0
                x_max_pad = 0
            else:
                x_pad = 32 - width % 32
                x_min_pad = int(x_pad / 2)
                x_max_pad = x_pad - x_min_pad

            img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)
            img=img[:,:,0]
            cv2.imwrite(os.path.join(new_root,fileobj),img)


if __name__=='__main__':
    pad('/home/paperspace/data/datasets/xinxueguan/SegmentationClass','/home/paperspace/data/datasets/unet_xinxueguan/SegmentationClass')
