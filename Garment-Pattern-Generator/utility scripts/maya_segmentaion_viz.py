"""
    Visualize mesh with segmentation lables.
    
    Tiny script to be executed within Maya Python environment, primaily designed for debug purposes
"""

import os
import numpy as np
import time

from numpy.lib.arraysetops import unique

# Maya
import maya.cmds as cmds

# My 
from mayaqltools import utils

# setup
base_path = 'D:/Data/my garments/'  # for all datasets  # Assuming no access to system.json file inside Maya
dataset = 'data_1600_skirt_4_panels_210525-12-46-54'
datapoint = 'skirt_4_panels_9VXDO0QR74'
tag = 'scan_imitation'  # 'scan_imitation'  'sim'

obj_path = os.path.join(base_path, dataset, datapoint, datapoint + '_' + tag + '.obj')
segmentation_path = os.path.join(base_path, dataset, datapoint, datapoint + '_' + tag + '_segmentation.txt')


# load geometry
garment = utils.load_file(os.path.join(obj_path))
num_verts = cmds.polyEvaluate(garment, vertex=True)

# load labels
with open(segmentation_path, 'r') as f:
    vert_labels = [line.rstrip() for line in f]  # remove \n
unique_labels = list(set(vert_labels))  # all unique labels available

print('Number of vertices: in mesh={}, Labels file={}'.format(num_verts, len(vert_labels)))
print(unique_labels)

start_time = time.time()

# group vertices by label
vertex_select_lists = dict.fromkeys(unique_labels)
for key in vertex_select_lists:
    vertex_select_lists[key] = []
for vert_idx in range(min(num_verts, len(vert_labels))):
    str_label = vert_labels[vert_idx]
    vert_addr = '{}.vtx[{}]'.format(garment, vert_idx)
    vertex_select_lists[str_label].append(vert_addr)

# Base colors
# https://www.schemecolor.com/bright-rainbow-colors.php
color_hex = ['FF0900', 'FF7F00', 'FFEF00', '00F11D', '0079FF', 'A800FF']
color_list = np.empty((len(color_hex), 3))
for idx in range(len(color_hex)):
    color_list[idx] = np.array([int(color_hex[idx][i:i + 2], 16) for i in (0, 2, 4)]) / 255.0

# color them
for label, str_label in enumerate(unique_labels):
    if str_label == 'stitch':
        color = np.zeros(3)
    else:
        # color selection with expansion if the list is too small
        factor, color_id = (label // len(color_list)) + 1, label % len(color_list)
        color = color_list[color_id] / factor  # gets darker the more labels there are

    print(len(vertex_select_lists[str_label]), num_verts)

    # color corresponding vertices
    if len(vertex_select_lists[str_label]) > 0:
        cmds.select(clear=True)
        cmds.select(vertex_select_lists[str_label])
        cmds.polyColorPerVertex(rgb=color.tolist())

cmds.select(clear=True)
cmds.setAttr(garment + '.displayColors', 1)
cmds.refresh()
print('Colorization: ', time.time() - start_time)
