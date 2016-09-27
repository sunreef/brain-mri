import numpy as np


class VoxelGrid:
    def __init__(self, img, size=16):
        self.voxelSize = size

        img_array = np.array(img.dataobj)

        size_x, size_y, size_z, dim = img.shape

        size_x /= size
        size_y /= size
        size_z /= size

        self.grid = np.zeros((size_x, size_y, size_z))

        for x in range(0, size_x):
            for y in range(0, size_y):
                for z in range(0, size_z):
                    self.grid[x][y][z] = np.sum(img_array[x * size: (x + 1) * size,y * size: (y + 1) * size,z * size: (z + 1) * size,0])

    def __str__(self):

        size_x, size_y, size_z = self.grid.shape
        result = "Voxel Grid:\n\t- X dimension: " + str(size_x) \
                 + "\n\t- Y dimension: " + str(size_y) \
                 + "\n\t- Z dimension: " + str(size_z) \
                 + "\n\t[\n"
        for x in range(0, size_x):
            result += "\t\t[\n"
            for y in range(0, size_y):
                result += "\t\t\t["
                for z in range(0, size_z):
                    result += str(self.grid[x][y][z])
                    if z < size_z - 1:
                        result += " "
                result += "]\n"
            result += "\t\t]\n"
        result += "\t]"

        return result

    def get_feature_vector(self):
        size_x, size_y, size_z = self.grid.shape
        return np.reshape(self.grid, size_x * size_y * size_z)
