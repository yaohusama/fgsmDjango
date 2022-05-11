import os
import numpy as np
from PIL import Image
import cv2

def img_pad(pil_file,wh):
    w, h,c= pil_file.shape
    fixed_size = wh  # 输出正方形图片的尺寸

    if h >= w:
        factor = h / float(fixed_size)
        new_w = int(w / factor)
        if new_w % 2 != 0:
            new_w -= 1
        pil_file = cv2.resize(pil_file,(new_w, fixed_size))
        pad_w = int((fixed_size - new_w) / 2)
        array_file = np.array(pil_file)
        # array_file = np.pad(array_file, ((0, 0), (pad_w, fixed_size-pad_w)), 'constant')
        array_file = cv2.copyMakeBorder(array_file, 0, 0, pad_w, fixed_size-new_w-pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        factor = w / float(fixed_size)
        new_h = int(h / factor)
        if new_h % 2 != 0:
            new_h -= 1
        pil_file = cv2.resize(pil_file,(fixed_size, new_h))
        pad_h = int((fixed_size - new_h) / 2)
        array_file = np.array(pil_file)
        # array_file = np.pad(array_file, ((pad_h, fixed_size-pad_h), (0, 0)), 'constant')
        array_file = cv2.copyMakeBorder(array_file,  pad_h, fixed_size - new_h-pad_h,0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # output_file = Image.fromarray(array_file)
    return array_file


