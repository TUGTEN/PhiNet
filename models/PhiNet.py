import torch
import cv2
import numpy as np
from torch import nn
import torch.nn.functional as F

resize_shape = [128, 256]


def PreProcess(image):
    nparr = np.fromstring(image.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (resize_shape[0], resize_shape[1]))
    img.astype(np.float64)
    img = np.transpose(img[np.newaxis, np.newaxis, :, :], (0, 1, 3, 2))

    return torch.from_numpy(img.copy()).float()
