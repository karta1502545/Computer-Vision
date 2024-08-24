import os
import imageio
import numpy as np
from absl import flags, app
import math

FLAGS = flags.FLAGS
flags.DEFINE_string('test_name_hard', 'hard_sonnet', 
                    'what set of shreads to load')


def load_imgs(name):
    file_names = os.listdir(os.path.join('shredded-images', name))
    file_names.sort()
    Is = []
    for f in file_names:
        I = imageio.v2.imread(os.path.join('shredded-images', name, f))
        Is.append(I)
    return Is


def solve(Is):
    '''
    :param Is: list of N images
    :return order: order list of N images
    :return offsets: offset list of N ordered images
    '''
    order = [10, 3, 15, 16, 13, 0, 11, 1, 2, 7, 8, 9, 5, 17, 4, 14, 6, 12]
    offsets = [43, 0, 7, 24, 51, 49, 52, 35, 48, 45, 17, 21, 27, 2, 38, 32, 31, 34]
    # We are returning the order and offsets that will work for 
    # hard_campus, you need to write code that works in general for any given
    # Is. Use the solution for hard_campus to understand the format for
    # what you need to return
    order, offsets = getOrder(Is)

    return order, offsets

def pairwise_distance(Is):
    '''
    :param Is: list of N images
    :return dist: pairwise distance matrix of N x N
    
    Given a N image stripes in Is, returns a N x N matrix dist which stores the
    distance between pairs of shreds. Specifically, dist[i,j] contains the
    distance when strip j is just to the left of strip i.
    '''
    
    dist = np.zeros((len(Is), len(Is)))
    pairwise_offsets = np.zeros((len(Is), len(Is)))
    for i in range(len(Is)):
        for j in range(len(Is)):
            if i != j:
                # record the largest diff's offset
                # (+: left is lower, -: left is higher)
                offsetOfMinDist, minDist = 0, float("inf")
                # j is left side, i is right side
                numOfPixel = 1
                left = Is[j][:, -numOfPixel:]
                right = Is[i][:, :numOfPixel]

                large, small = left, right
                isOffsetMinus = True
                if left.shape[0] < right.shape[0]:
                    large, small = right, left
                    isOffsetMinus = False
                
                l, L = small.shape[0], large.shape[0]
                ofst = int(0.2 * L)
                for k in range(ofst, -(L+ofst-l), -1):
                    FragOfLarge, FragOfSmall = None, None
                    if k > 0: # small on top of large
                        FragOfSmall = small[-(l-k):]
                        FragOfLarge = large[:len(FragOfSmall)]
                    elif k < l - L: # small below large
                        FragOfLarge = large[-k:]
                        FragOfSmall = small[:len(FragOfLarge)]
                    else: # small in the middle of large
                        FragOfLarge = large[-k:-k+l]
                        FragOfSmall = small

                    # sum of squared distance
                    nowDiff = np.sum(np.square(FragOfSmall - FragOfLarge))
                    nowDiff = np.sqrt(nowDiff / FragOfSmall.shape[0])

                    if minDist > nowDiff:
                        minDist = nowDiff
                        offsetOfMinDist = k
                        offsetOfMinDist *= -1 if isOffsetMinus else 1
                
                # In the above loop:
                # 1. calculate distance for each pair (and do mean)
                # 2. update (the smallest) dist (from each pair), and its according offset(+/-) to pairwise_offsets

                dist[i, j] = minDist
                pairwise_offsets[i, j] = int(offsetOfMinDist)
    # print(dist)
    # print(pairwise_offsets)

    return dist, pairwise_offsets

def getOrder(Is):
    dist, pairwise_offsets = pairwise_distance(Is)

    inds = np.arange(len(Is))
    # run greedy matching
    order = [0]
    for i in range(len(Is) - 1):
        d1 = np.min(dist[0, 1:])
        d2 = np.min(dist[1:, 0])
        if d1 < d2:
            ind = np.argmin(dist[0, 1:]) + 1
            order.insert(0, inds[ind])
            dist[0, :] = dist[ind, :]
            dist = np.delete(dist, ind, 0)
            dist = np.delete(dist, ind, 1)
            inds = np.delete(inds, ind, 0)
        else:
            ind = np.argmin(dist[1:, 0]) + 1
            order.append(inds[ind])
            dist[:, 0] = dist[:, ind]
            dist = np.delete(dist, ind, 0)
            dist = np.delete(dist, ind, 1)
            inds = np.delete(inds, ind, 0)
    
    offsets = [0 for i in range(len(Is))]
    for k in range(1, len(Is)):
        # j is left side, i is right side
        j, i = order[k-1], order[k]
        offsets[k] = offsets[k-1] + int(pairwise_offsets[i][j])
        # offsets[k] = int(pairwise_offsets[i][j])
    
    minOfst = -min(offsets)
    offsets = [x + minOfst for x in offsets]

    # print("--------")
    # print(order)
    # print("--------")
    # print(offsets)
    # print("--------")

    return order, offsets


def composite(Is, order, offsets):
    Is = [Is[o] for o in order]
    strip_width = 1
    W = np.sum([I.shape[1] for I in Is]) + len(Is) * strip_width
    H = np.max([I.shape[0] + o for I, o in zip(Is, offsets)])
    H = int(H)
    W = int(W)
    I_out = np.ones((H, W, 3), dtype=np.uint8) * 255
    w = 0
    for I, o in zip(Is, offsets):
        I_out[o:o + I.shape[0], w:w + I.shape[1], :] = I
        w = w + I.shape[1] + strip_width
    return I_out

def main(_):
    Is = load_imgs(FLAGS.test_name_hard)
    order, offsets = solve(Is)
    I = composite(Is, order, offsets)
    import matplotlib.pyplot as plt
    plt.imshow(I)
    plt.show()

if __name__ == '__main__':
    app.run(main)
