import os
import numpy as np
import nibabel as nib

from voxel_grid import VoxelGrid

grid_size = 16;

target_data = ""
with open('../data/set_train/targets.csv', 'r') as fo:
    target_data = fo.read()

numbers = target_data.split('\n')
numbers.pop(len(numbers) - 1)

vectorized_int = np.vectorize(int)
target_ages = vectorized_int(numbers)

for i in range(1, 279):
    img = nib.load("../data/set_train/train_" + str(i) + ".nii")
    grid = VoxelGrid(img, grid_size)

    save_path = "../data/features/grid_size_" + str(grid_size)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(save_path + "/feature_vector_" + str(i), grid.get_feature_vector())
    print("Saved feature vector #" + str(i))
