import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_gradient_magnitude
from scipy.ndimage.filters import gaussian_filter
import cv2

def compute_edges_dxdy(I_origin):
  """Returns the norm of dx and dy as the edge response function."""
  I_origin = cv2.cvtColor(I_origin, cv2.COLOR_BGR2GRAY)
  I_origin = I_origin.astype(np.float32)/255.

  ## original code
  # dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same', boundary='symm')
  # dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same', boundary='symm')
  # mag = np.sqrt(dx**2 + dy**2)
  # mag = mag / 1.5

  scales = [1,2,3,4]
  mag_scale = np.zeros(I_origin.shape, dtype=I_origin.dtype)
  for scale in scales:
    I = I_origin.copy()
    if scale != 1:
      I = cv2.resize(I, (0,0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_LINEAR)

    ## Smoothing
    sigma = 1
    # mag = gaussian_gradient_magnitude(I, sigma=sigma)
    I = gaussian_filter(I,sigma=sigma)

    ## gradient
    gradient = np.gradient(I)
    mag = np.sqrt(gradient[0] ** 2 + gradient[1] ** 2)
    direction = np.arctan2(gradient[1], gradient[0])

    ## fine grain the edge using non maximum supression
    mag_max = np.zeros(mag.shape, dtype=mag.dtype)

    for i in range(I.shape[1]):
      for j in range(I.shape[0]):
        theta = direction[j, i]
        theta = (theta + np.pi) % np.pi
        n1, n2 = 0, 0
        # TODO: use interpolation
          
        if (n1 == 0 and n2 == 0) or max(mag[j][i], n1, n2) == mag[j][i]:
          # find the peak, so store it
          mag_max[j][i] = mag[j][i]

        mag = cv2.resize(mag, (I_origin.shape[1], I_origin.shape[0]), interpolation=cv2.INTER_LINEAR)
        mag_scale = np.maximum(mag_scale, mag)
  
  mag_scale = mag_scale * 255.
  mag_scale = np.clip(mag_scale, 0, 255)
  mag_scale = mag_scale.astype(np.uint8)
  return mag_scale