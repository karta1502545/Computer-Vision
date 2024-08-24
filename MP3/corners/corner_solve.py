import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
import cv2

def compute_corners(I):
  # Currently this code proudces a dummy corners and a dummy corner response
  # map, just to illustrate how the code works. Your task will be to fill this
  # in with code that actually implements the Harris corner detector. You
  # should return th ecorner response map, and the non-max suppressed corners.
  # Input:
  #   I: input image, H x W x 3 BGR image
  # Output:
  #   response: H x W response map in uint8 format
  #   corners: H x W map in uint8 format _after_ non-max suppression. Each
  #   pixel stores the score for being a corner. Non-max suppressed pixels
  #   should have a low / zero-score.
  
  ## original code: the corners are random generated
  # rng = np.random.RandomState(int(np.sum(I.astype(np.float32))))
  # sz = I.shape[:2]
  # corners = rng.rand(*sz)
  # corners = np.clip((corners - 0.95)/0.05, 0, 1)
  # response = scipy.ndimage.gaussian_filter(corners, 4, order=0, output=None,
  #                                          mode='reflect')

  # Harris
  sigma = 2
  I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
  # Iy, Ix = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3), cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
  Iy, Ix = np.gradient(I)
  Ixx = gaussian_filter(Ix**2, sigma)
  Iyy = gaussian_filter(Iy**2, sigma)
  Ixy = gaussian_filter(Ix*Iy, sigma)

  alpha = 0.06 # TODO: hyperparameter 0.04-0.06

  # R = det(M) - alpha * (trace(M)**2)
  # print("Hello")
  R = (Ixx*Iyy - Ixy**2) - alpha * ((Ixx + Iyy) ** 2)
  threshold = 0.015 * R.max()
  # print(f"threshold = {threshold}")
  corners = np.where(R > threshold, R, 0)
  print(sigma, threshold)

  k = 5 # window size
  # print(f"k={k}, aplha={alpha}, threshold={0.01}")
  # nmsCorners = np.zeros(corners.shape, dtype=corners.dtype)
  # for i in range(k//2, corners.shape[0] - k//2):
  #   for j in range(k//2, corners.shape[1] - k//2):
  #     window = corners[(i-k//2):(i+k//2+1), (j-k//2):(j+k//2+1)]
  #     # get max of k by k windows
  #     maxNum = np.max(window)
  #     # if corners[i][j] != max, then it should be discarded
  #     if corners[i, j] == maxNum:
  #       nmsCorners[i,j] = corners[i,j]
      
  # corners = nmsCorners # TODO: revise it to clean code
  corners = corners * 255.
  corners = np.clip(corners, 0, 255)
  corners = corners.astype(np.uint8)
  
  response = R * 255.
  response = np.clip(response, 0, 255)
  response = response.astype(np.uint8)
  
  return response, corners