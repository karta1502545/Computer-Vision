import numpy as np
from generate_scene import get_ball
from generate_scene_bunny import get_bunny
import matplotlib.pyplot as plt

def normalize(v):
  norms = np.linalg.norm(v, axis=-1, keepdims=True)
  v = v / norms
  return v

# specular exponent
k_e = 50

def render(Z, N, A, S, 
           point_light_loc, point_light_strength,
           directional_light_dirn, directional_light_strength,
           ambient_light, k_e):
  # To render the images you will need the camera parameters, you can assume
  # the following parameters. (cx, cy) denote the center of the image (point
  # where the optical axis intersects with the image, f is the focal length.
  # These parameters along with the depth image will be useful for you to
  # estimate the 3D points on the surface of the sphere for computing the
  # angles between the different directions.
  h, w = A.shape
  cx, cy = w / 2, h /2
  f = 128. # related to Vr (x,y,f)

  # N = normalize(N)

  # Ambient Term
  I = A * ambient_light

  Vi, Li = None, None
  if point_light_strength[0] != 0:
    Li = point_light_strength

    i, j = np.tile(np.arange(w), (h, 1)), np.tile(np.arange(h), (w, 1)).T
    # print(i.shape)
    # x = np.arange(w)
    # y = np.arange(h)
    # i, j = np.meshgrid(x, y)
    # print(i.shape)
    # print("------------")

    points3D = np.stack((Z*(i-cx)/f, Z*(j-cy)/f, Z), axis = -1)
    Vi = point_light_loc - points3D
    Vi = normalize(Vi)
  else: # directional_light_strength != 0
    Li = directional_light_strength

    Vi = np.array(directional_light_dirn)
    Vi = np.tile(Vi, (h, w, 1))
    Vi = normalize(Vi)
  
  # Diffuse Term
  i_p = np.einsum('ijk,ijk->ij', N, Vi)
  i_p = np.where(i_p < 0, 0, i_p)
  D = A * Li * i_p
  I += D

  # Specular Term

  # \hat{Vr}[i][j] = (cx-i, cy-j, z), given the center of scene = (cx,cy)
  i, j = np.tile(np.arange(w), (h, 1)), np.tile(np.arange(h), (w, 1)).T
  # print("-------------")
  # print(i)
  # x = np.arange(w)
  # y = np.arange(h)
  # i, j = np.meshgrid(x, y)
  # print(i)
  # print("-------------")

  Vr = -np.stack((Z*(i-cx)/f, Z*(j-cy)/f, Z), axis = -1)
  Vr = normalize(Vr)

  # \hat{Si}
  Si = np.einsum('ijk,ijk->ij', N, Vi)
  Si = Si[..., np.newaxis]
  Si = 2 * Si * N - Vi
  Si = normalize(Si)
  
  S_term = np.einsum('ijk,ijk->ij', Vr, Si)
  S_term = np.where(S_term < 0, 0, S_term)
  S_term = S * Li * np.power(S_term, k_e)
  I += S_term

  I = np.minimum(I, 1)*255
  I = I.astype(np.uint8)
  I = np.repeat(I[:,:,np.newaxis], 3, axis=2)
  return I

def main():
  for specular in [True, False]:
    # get_ball function returns:
    # - Z (depth image: distance to scene point from camera center, along the
    # Z-axis)
    # - N is the per pixel surface normals (N[:,:,0] component along X-axis
    # (pointing right), N[:,:,1] component along Y-axis (pointing down),
    # N[:,:,2] component along Z-axis (pointing into the scene)),
    # - A is the per pixel ambient and diffuse reflection coefficient per pixel,
    # - S is the per pixel specular reflection coefficient.

    # Z, N, A, S = get_ball(specular=specular)
    Z, N, A, S = get_bunny(specular=specular)

    # Strength of the ambient light.
    ambient_light = 0.5
    
    # For the following code, you can assume that the point sources are located
    # at point_light_loc and have a strength of point_light_strength. For the
    # directional light sources, you can assume that the light is coming _from_
    # the direction indicated by directional_light_dirn (\hat{v}_i = directional_light_dirn), and with strength
    # directional_light_strength. The coordinate frame is centered at the
    # camera, X axis points to the right, Y-axis point down, and Z-axis points
    # into the scene.
    
    # Case I: No directional light, only point light source that moves around
    # the object.
    point_light_strength = [1.5]
    directional_light_dirn = [[1, 0, 0]]
    directional_light_strength = [0.0]
    
    fig, axes = plt.subplots(4, 4, figsize=(15,10))
    axes = axes.ravel()[::-1].tolist()
    for theta in np.linspace(0, np.pi*2, 16): 
      point_light_loc = [[10*np.cos(theta), 10*np.sin(theta), -3]]
      I = render(Z, N, A, S, point_light_loc, point_light_strength, 
                 directional_light_dirn, directional_light_strength,
                 ambient_light, k_e)
      ax = axes.pop()
      ax.imshow(I)
      ax.set_axis_off()
    plt.savefig(f'specular{specular:d}_move_point.png', bbox_inches='tight')
    plt.close()

    # Case II: No point source, just a directional light source that moves
    # around the object.
    point_light_loc = [[0, -10, 2]]
    point_light_strength = [0.0]
    directional_light_strength = [2.5]
    
    fig, axes = plt.subplots(4, 4, figsize=(15,10))
    axes = axes.ravel()[::-1].tolist()
    for theta in np.linspace(0, np.pi*2, 16): 
      directional_light_dirn = [np.array([np.cos(theta), np.sin(theta), .1])]
      directional_light_dirn[0] = \
        directional_light_dirn[0] / np.linalg.norm(directional_light_dirn[0])
      I = render(Z, N, A, S, point_light_loc, point_light_strength, 
                 directional_light_dirn, directional_light_strength,
                 ambient_light, k_e) 
      ax = axes.pop()
      ax.imshow(I)
      ax.set_axis_off()
    plt.savefig(f'specular{specular:d}_move_direction.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
  main()
