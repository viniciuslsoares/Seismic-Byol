import torchvision.transforms as transforms
import random

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            t = transforms.GaussianBlur(kernel_size=5,sigma=sigma)
            return t(img)
        else:
            return img
