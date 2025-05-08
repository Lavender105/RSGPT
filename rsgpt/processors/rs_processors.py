"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

from rsgpt.common.registry import registry
from rsgpt.processors.base_processor import BaseProcessor
from rsgpt.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class RSImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)


@registry.register_processor("rs_image_train")
class RSImageTrainProcessor(RSImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
        )


@registry.register_processor("rs_image_eval")
class RSImageEvalProcessor(RSImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)