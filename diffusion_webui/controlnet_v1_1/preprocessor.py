import gc

import numpy as np
import PIL.Image
import torch
from controlnet_aux import (CannyDetector, ContentShuffleDetector, HEDdetector,
                            LineartAnimeDetector, LineartDetector,
                            MidasDetector, MLSDdetector, NormalBaeDetector,
                            OpenposeDetector, PidiNetDetector)
from controlnet_aux.util import HWC3

from .cv_utils import resize_image
from .depth_estimator import DepthEstimator
from .image_segmentor import ImageSegmentor


class Preprocessor:
    MODEL_ID = 'lllyasviel/Annotators'

    def __init__(self):
        self.model = None
        self.name = ''

    def load(self, name: str) -> None:
        if name == self.name:
            return
        if name == 'HED':
            self.model = HEDdetector.from_pretrained(self.MODEL_ID,cache_dir = './anno_model')
        elif name == 'Midas':
            self.model = MidasDetector.from_pretrained(self.MODEL_ID,cache_dir = './anno_model')
        elif name == 'MLSD':
            self.model = MLSDdetector.from_pretrained(self.MODEL_ID,cache_dir = './anno_model')
        elif name == 'Openpose':
            self.model = OpenposeDetector.from_pretrained(self.MODEL_ID,cache_dir = './anno_model')
        elif name == 'PidiNet':
            self.model = PidiNetDetector.from_pretrained(self.MODEL_ID,cache_dir = './anno_model')
        elif name == 'NormalBae':
            self.model = NormalBaeDetector.from_pretrained(self.MODEL_ID,cache_dir = './anno_model')
        elif name == 'Lineart':
            self.model = LineartDetector.from_pretrained(self.MODEL_ID,cache_dir = './anno_model')
        elif name == 'LineartAnime':
            self.model = LineartAnimeDetector.from_pretrained(self.MODEL_ID,cache_dir = './anno_model')
        elif name == 'Canny':
            self.model = CannyDetector()
        elif name == 'ContentShuffle':
            self.model = ContentShuffleDetector()
        elif name == 'DPT':
            self.model = DepthEstimator()
        elif name == 'UPerNet':
            self.model = ImageSegmentor()
        else:
            raise ValueError
        torch.cuda.empty_cache()
        gc.collect()
        self.name = name

    def __call__(self, image: PIL.Image.Image, **kwargs) -> PIL.Image.Image:
        if self.name == 'Canny':
            if 'detect_resolution' in kwargs:
                detect_resolution = kwargs.pop('detect_resolution')
                image = np.array(image)
                image = HWC3(image)
                image = resize_image(image, resolution=detect_resolution)
            image = self.model(image, **kwargs)
            return PIL.Image.fromarray(image)
        elif self.name == 'Midas':
            detect_resolution = kwargs.pop('detect_resolution', 512)
            image_resolution = kwargs.pop('image_resolution', 512)
            image = np.array(image)
            image = HWC3(image)
            image = resize_image(image, resolution=detect_resolution)
            image = self.model(image, **kwargs)
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            return PIL.Image.fromarray(image)
        else:
            return self.model(image, **kwargs)
