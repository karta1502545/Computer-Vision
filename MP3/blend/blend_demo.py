# Code from Saurabh Gupta
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags

from blend_solve import blend

FLAGS = flags.FLAGS
flags.DEFINE_string('im1', 'face1.jpeg', 'image 1')
flags.DEFINE_string('im2', 'face2.jpeg', 'image 2')
flags.DEFINE_string('mask', 'mask.png', 'mask image')
flags.DEFINE_string('out_name', 'output.png', 'output image name')

def main(_):
  I1 = cv2.imread(FLAGS.im1)
  I2 = cv2.imread(FLAGS.im2)
  mask = cv2.imread(FLAGS.mask)
  print(I1.shape, I2.shape)
  
  out = blend(I1, I2, mask)
  # save image
  cv2.imwrite(FLAGS.out_name, out)

if __name__ == '__main__':
  app.run(main)
