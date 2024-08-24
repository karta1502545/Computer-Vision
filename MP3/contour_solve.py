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
    base_theta = np.pi / 8
    theta_list = [base_theta, base_theta*3, base_theta*5, base_theta*7]

    for i in range(I.shape[1]):
      for j in range(I.shape[0]):
        theta = direction[j, i]
        theta = (theta + np.pi) % np.pi
        n1, n2 = 0, 0
        if (0 <= theta < theta_list[0] or theta_list[3] <= theta < np.pi):
          # up-down
          if (i-1 >=0 and i+1 < I.shape[1]):
            n1, n2 = mag[j][i-1], mag[j][i+1]
        elif (theta_list[0] <= theta <theta_list[1]):
          # diagonal (negative slope)
          if (i-1 >=0 and i+1 < I.shape[1] and j-1 >=0 and j+1 < I.shape[0]):
            n1, n2 = mag[j-1][i-1], mag[j+1][i+1]
        elif (theta_list[1] <= theta <theta_list[2]):
          # left-right
          if (j-1 >=0 and j+1 < I.shape[0]):
            n1, n2 = mag[j+1][i], mag[j-1][i]
        else:
          # diagonal (positive slope)
          if (i-1 >=0 and i+1 < I.shape[1] and j-1 >=0 and j+1 < I.shape[0]):
            n1, n2 = mag[j-1][i+1], mag[j+1][i-1]
          
        if (n1 == 0 and n2 == 0) or max(mag[j][i], n1, n2) == mag[j][i]:
          # find the peak, so store it
          mag_max[j][i] = mag[j][i]

        mag = cv2.resize(mag, (I_origin.shape[1], I_origin.shape[0]), interpolation=cv2.INTER_LINEAR)
        mag_scale = np.maximum(mag_scale, mag)
  
  mag_scale = mag_scale * 255.
  mag_scale = np.clip(mag_scale, 0, 255)
  mag_scale = mag_scale.astype(np.uint8)
  return mag_scale







'''
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_gradient_magnitude
from scipy.ndimage.filters import gaussian_filter
import cv2
from math import ceil, floor

def interpolate(mag, x, y):
    x1, y1 = int(x), int(y)
    x2, y2 = x1 + 1, y1 + 1
    
    a = x - x1
    b = y - y1
    
    if x2 >= mag.shape[1]:
        x2 = x1
    if y2 >= mag.shape[0]:
        y2 = y1
    
    value = (1 - a) * (1 - b) * mag[y1, x1] + a * (1 - b) * mag[y1, x2] + \
            (1 - a) * b * mag[y2, x1] + a * b * mag[y2, x2]
    return value

def compute_edges_dxdy(I_origin):
  """Returns the norm of dx and dy as the edge response function."""
  I_origin = cv2.cvtColor(I_origin, cv2.COLOR_BGR2GRAY)
  I_origin = I_origin.astype(np.float32)/255.

  ## original code
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

    for i in range(1, I.shape[1]-1):
      for j in range(1, I.shape[0]-1):
        theta = direction[j, i]
        theta = (theta + np.pi) % np.pi
        n1, n2 = 0, 0
        
        # TODO: interpolation
        dx = np.cos(theta)
        dy = np.sin(theta)

        forward_x = i + dx
        forward_y = j + dy
        reverse_x = i - dx
        reverse_y = j - dy

        n1 = interpolate(mag, forward_x, forward_y)
        n2 = interpolate(mag, reverse_x, reverse_y)
        
        if (n1 == 0 and n2 == 0) or max(mag[j][i], n1, n2) == mag[j][i]:
          # find the peak, so store it
          mag_max[j][i] = mag[j][i]

        mag = cv2.resize(mag, (I_origin.shape[1], I_origin.shape[0]), interpolation=cv2.INTER_LINEAR)
        mag_scale = np.maximum(mag_scale, mag)
  
  mag_scale = mag_scale * 255.
  mag_scale = np.clip(mag_scale, 0, 255)
  mag_scale = mag_scale.astype(np.uint8)
  return mag_scale



'''