import colorsys
import copy
import numpy as np
import cv2
import os
import random
from utility import *


attri_name = ['pattern', 'major_color', 'necktie', 'collar', 'scarf', 'sleeve', 'neckline', 'clothing', 'jacket', 'hat', \
              'glass', 'layer', 'necktie_color', 'necktie_pattern', 'hair_color', 'hair_length', 'hair_colorNN']
attri_dict = [
        {'solid': 0, 'graphics' : 1, 'striped' : 2, 'floral' : 3, 'plaid' : 4, 'spotted' : 5}, # clothing_pattern
        {'black' : 0, 'white' : 1, 'more color' : 2, 'blue' : 3, 'gray' : 4, 'red' : 5,
                'pink' : 6, 'green' : 7, 'yellow' : 8, 'brown' : 9, 'purple' : 10, 'orange' : 11,
                'cyan' : 12, 'dark blue' : 13}, # major_color
        {'necktie no': 0, 'necktie yes' : 1}, # wearing_necktie
        {'collar no': 0, 'collar yes' : 1}, # collar_presence
        {'scarf no': 0, 'scarf yes' : 1}, # wearing_scarf
        {'long sleeve' : 0, 'short sleeve' : 1, 'no sleeve' : 2}, # sleeve_length
        {'round' : 0, 'folded' : 1, 'v-shape' : 2}, # neckline_shape
        {'shirt' : 0, 'outerwear' : 1, 't-shirt' : 2, 'dress' : 3,
            'tank top' : 4, 'suit' : 5, 'sweater' : 6}, # clothing_category
        {'jacket no': 0, 'jacket yes' : 1}, # wearing_jacket
        {'hat no': 0, 'hat yes' : 1}, # wearing_hat
        {'glasses no': 0, 'glasses yes' : 1}, # wearing_glasses
        {'one layer': 0, 'more layer' : 1}, # multiple_layers
        {'black' : 0, 'white' : 1, 'more color' : 2, 'blue' : 3, 'gray' : 4, 'red' : 5,
                'pink' : 6, 'green' : 7, 'yellow' : 8, 'brown' : 9, 'purple' : 10, 'orange' : 11,
                'cyan' : 12, 'dark blue' : 13}, # necktie_color
        {'solid' : 0, 'striped' : 1, 'spotted' : 2}, # necktie_pattern
        {'black' : 0, 'white': 1, 'blond' : 2, 'brown' : 3, 'gray' : 4}, # hair_color
        {'long' : 0, 'medium' : 1, 'short' : 2, 'bald' : 3}, # hair_longth
        {'black' : 0, 'white': 1, 'blond' : 2, 'brown' : 3, 'gray' : 4} # hair_color
]
attri_num = len(attri_name)
attri_dict_inv = []
for d in attri_dict:
    d_inv = {}
    for k, v in d.items():
        d_inv[v] = k
    attri_dict_inv.append(d_inv)


def compress_images(images):
    images_avg = []
#     N = min(len(images), 10000)
#     images = images[:N]
    for img in images:
        images_avg.append(np.average(img.reshape((-1, 3)), axis=0))
    images_avg = np.array(images_avg)
    return images_avg

# sort images color by Nearest Neighbor
def NN(A, start):
    """Nearest neighbor algorithm.
    A is an NxN array indicating distance between N locations
    start is the index of the starting location
    Returns the path and cost of the found solution
    """
    path = [start]
    cost = 0
    N = A.shape[0]
    mask = np.ones(N, dtype=bool)  # boolean values indicating which 
                                   # locations have not been visited
    mask[start] = False

    for i in range(N-1):
        last = path[-1]
        next_ind = np.argmin(A[last][mask]) # find minimum of remaining locations
        next_loc = np.arange(N)[mask][next_ind] # convert to original location
        path.append(next_loc)
        mask[next_loc] = False
        cost += A[last, next_loc]

    return path, cost

def sort_images_by_NN(images):
    images_avg = compress_images(images)
    N = len(images)
    A = np.zeros((N, N))
    for i in range(0, N):
        dist = images_avg - images_avg[i]
        A[i] = np.linalg.norm(dist, axis=1)

    # Nearest neighbour algorithm
    path, _ = NN(A, 1000)

    # Final array
    images_sort = []
    for i in path:
        images_sort.append(images[i])
    return images_sort


# sort images color by step sort
def step (r,g,b, repetitions=1):
    lum = math.sqrt( .241 * r + .691 * g + .068 * b )

    h, s, v = colorsys.rgb_to_hsv(r,g,b)

    h2 = int(h * repetitions) + 1
    if s < 0.2:
        h2 = 0
    elif s > 0.8:
        h2 = repetitions
    v2 = int(v * repetitions)
    s2 = int(s * repetitions)

#     if h2 % 2 == 1:
#         v2 = repetitions - v2
#         lum2 = repetitions - lum2
#         s2 = repetitions - s2

    return (h2, s2, v2)
def swap(l, i, j):
    tmp = l[i]
    l[i] = l[j]
    l[j] = tmp

def sort_images_by_step(images):
    images_avg = compress_images(images)
    images_hsv = [step(pixel[2], pixel[1], pixel[0], 8) for pixel in images_avg]
    images_idx = np.arange(len(images))
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            if images_hsv[i] > images_hsv[j]:
                swap(images_hsv, i, j)
                swap(images_idx, i, j)
    images_sort = []
    for idx in images_idx:
        images_sort.append(images[idx])
    return images_sort