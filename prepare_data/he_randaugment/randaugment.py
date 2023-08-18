"""
The he_randaugment module comes from https://github.com/DIAGNijmegen/pathology-he-auto-augment
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
This repository is build upon RandAugment implementation
https://arxiv.org/abs/1909.13719 published here
https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py

"""
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AutoAugment and RandAugment policies for enhanced image preprocessing.
AutoAugment Reference: https://arxiv.org/abs/1805.09501
RandAugment Reference: https://arxiv.org/abs/1909.13719
"""
import inspect
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from .augmenters.color.hsbcoloraugmenter import HsbColorAugmenter
from .augmenters.color.hedcoloraugmenter import HedColorAugmenter

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.0


def hsv(image, factor):
    # print('image',image.shape)
    image = np.transpose(image, [2, 0, 1])
    augmentor = HsbColorAugmenter(
        hue_sigma_range=(-factor, factor),
        saturation_sigma_range=(-factor, factor),
        brightness_sigma_range=(0, 0),
    )
    # To select a random magnitude value between -factor:factor, if commented the m value will be constant
    augmentor.randomize()
    return np.transpose(augmentor.transform(image), [1, 2, 0])


def hed(image, factor):
    # print('image',image.shape)
    image = np.transpose(image, [2, 0, 1])
    augmentor = HedColorAugmenter(
        haematoxylin_sigma_range=(-factor, factor),
        haematoxylin_bias_range=(-factor, factor),
        eosin_sigma_range=(-factor, factor),
        eosin_bias_range=(-factor, factor),
        dab_sigma_range=(-factor, factor),
        dab_bias_range=(-factor, factor),
        cutoff_range=(0.15, 0.85),
    )
    ##To select a random magnitude value between -factor:factor, if commented the m value will be constant
    augmentor.randomize()
    return np.transpose(augmentor.transform(image), [1, 2, 0])


def color(image, factor):
    """Equivalent of PIL Color."""
    image = Image.fromarray(image)
    image = ImageEnhance.Color(image).enhance(factor)
    return np.asarray(image)


def contrast(image, factor):
    """Equivalent of PIL Contrast."""
    image = Image.fromarray(image)
    image = ImageEnhance.Contrast(image).enhance(factor)
    return np.asarray(image)


def brightness(image, factor):
    """Equivalent of PIL Brightness."""
    image = Image.fromarray(image)
    image = ImageEnhance.Brightness(image).enhance(factor)
    return np.asarray(image)


def sharpness(image, factor):
    """Implements Sharpness function from PIL using TF ops."""
    image = Image.fromarray(image)
    image = ImageEnhance.Sharpness(image).enhance(factor)
    return np.asarray(image)


def equalize(image):
    """Implements Equalize function from PIL using TF ops."""
    image = Image.fromarray(image)
    image = ImageOps.equalize(image)
    return np.asarray(image)


NAME_TO_FUNC = {
    "Hsv": hsv,
    "Hed": hed,
    "Equalize": equalize,
    "Color": color,
    "Contrast": contrast,
    "Brightness": brightness,
    "Sharpness": sharpness,
}


def _enhance_level_to_arg(level):
    return ((level / _MAX_LEVEL) * 1.8 + 0.1,)


def _enhance_level_to_arg_hsv(level):
    return (level * 0.03,)


def _enhance_level_to_arg_hed(level):
    return (level * 0.03,)


def level_to_arg(_):
    return {
        "Hsv": _enhance_level_to_arg_hsv,
        "Hed": _enhance_level_to_arg_hed,
        "Equalize": lambda level: (),
        "Color": _enhance_level_to_arg,
        "Contrast": _enhance_level_to_arg,
        "Brightness": _enhance_level_to_arg,
        "Sharpness": _enhance_level_to_arg,
    }


def _parse_policy_info(
    name, prob, level, replace_value, augmentation_hparams, magnitude
):
    """Return the function that corresponds to `name` and update `level` param."""
    if name == "Hsv":
        func = NAME_TO_FUNC[name]
        args = level_to_arg(augmentation_hparams)[name](magnitude)
    if name == "Hed":
        func = NAME_TO_FUNC[name]
        args = level_to_arg(augmentation_hparams)[name](magnitude)
    else:
        func = NAME_TO_FUNC[name]
        args = level_to_arg(augmentation_hparams)[name](level)

    # Check to see if prob is passed into function. This is used for operations
    # where we alter bboxes independently.
    # pytype:disable=wrong-arg-types
    if "prob" in inspect.getargspec(func)[0]:
        args = tuple([prob] + list(args))
    # pytype:enable=wrong-arg-types

    # Add in replace arg if it is required for the function that is being called.
    # pytype:disable=wrong-arg-types
    if "replace" in inspect.getargspec(func)[0]:
        # Make sure replace is the final argument
        assert "replace" == inspect.getargspec(func)[0][-1]
        args = tuple(list(args) + [replace_value])
    # pytype:enable=wrong-arg-types

    return (func, prob, args)


def distort_image_with_randaugment(image, num_layers, magnitude):
    """Applies the RandAugment policy to `image`.
    RandAugment is from the paper https://arxiv.org/abs/1909.13719,
    Args:
        image: `Tensor` of shape [height, width, 3] representing an image.
        num_layers: Integer, the number of augmentation transformations to apply
            sequentially to an image. Represented as (N) in the paper. Usually best
            values will be in the range [1, 3].
        magnitude: Integer, shared magnitude across all augmentation operations.
            Represented as (M) in the paper. Usually best values are in the range
            [1, 10].
        ra_type: List of augmentations to use
    Returns:
        The augmented version of `image`.
    """

    # PIL -> numpy
    image = np.asarray(image)

    replace_value = (128, 128, 128)  # [128] * 3
    augmentation_hparams = dict(cutout_const=40, translate_const=10)

    available_ops = [
        "Brightness",
        "Sharpness",
        "Color",
        "Contrast",
        "Equalize",
        "Hsv",
        "Hed",
    ]

    for _ in range(num_layers):
        op_to_select = np.random.randint(low=0, high=len(available_ops))
        random_magnitude = np.random.uniform(low=0, high=magnitude)
        for i, op_name in enumerate(available_ops):
            prob = np.random.uniform(low=0.2, high=0.8)
            func, _, args = _parse_policy_info(
                op_name,
                prob,
                random_magnitude,
                replace_value,
                augmentation_hparams,
                magnitude,
            )

            if i == op_to_select:
                selected_func = func
                selected_args = args
                image = selected_func(image, *selected_args)

            else:
                image = image

    return Image.fromarray(image)
