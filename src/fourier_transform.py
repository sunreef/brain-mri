import numpy as np

class FourierTransform:

    def __init__(self, img_array):
        self.output_array = np.fft.fftn(img_array)

    def return_output_array(self):
        return self.output_array