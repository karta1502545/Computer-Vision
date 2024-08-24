import numpy as np
from cv2 import GaussianBlur
from skimage.filters import gaussian, difference_of_gaussians
import cv2
# import datetime

def blend(im1, im2, mask):
  mask = mask / 255.
  # out = im1 * mask + (1-mask) * im2
  sigma = 8 # TODO: hyperparameter
  out = np.zeros(im1.shape)
  # print(datetime.datetime.now())
  # apply gaussian on im1, im2, mask
  for i in range(6):
    low_s = sigma * (2**(i-1)) if i > 0 else 0
    high_s = sigma * (2**i)

    im1_lap = difference_of_gaussians(im1, low_s, high_s, mode='reflect')
    im2_lap = difference_of_gaussians(im2, low_s, high_s, mode='reflect')

    mask_gaussian = gaussian(mask, high_s, channel_axis=-1)
    out += im1_lap * mask_gaussian + im2_lap * (1 - mask_gaussian)
    
  out = (out - np.min(out)) / (np.max(out) - np.min(out)) 
  out = out * 255.
  out = out.astype('uint8')

  return out
