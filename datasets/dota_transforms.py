# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random
import math
import cv2
import PIL
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import copy
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from util.misc import interpolate
import numpy as np
import BboxToolkit as bt
from functools import partial

def crop(image, target, region, thred=0.3,):

    cropped_image = F.crop(image, *region)

    tgt = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    tgt["size"] = torch.tensor([h, w])

    fields = ["labels", "theta"]
    boxes, theta = target["boxes"], target["theta"]
    bbox1 = torch.cat([box_xyxy_to_cxcywh(boxes), theta], dim = 1).numpy()
    bbox2 = np.array([[i+h*0.5, j+w*0.5, h, w, 0.0]])
    boxes_overlap = bt.bbox_overlaps(bbox1, bbox2).squeeze()
    keep = np.where(boxes_overlap >= thred)[0]
    cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
    cropped_polys = target['polys'] - torch.as_tensor([j, i, j, i, j, i, j, i])
    tgt["boxes"] = cropped_boxes[keep, :]
    tgt["polys"] = cropped_polys[keep, :]

    for field in fields:
        tgt[field] = target[field][keep]

    return cropped_image, tgt


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    tgt = target.copy()
    boxes = target["boxes"]
    boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
    polys = target["polys"]
    polys = polys[:, [2, 3, 0, 1, 6, 7, 4, 5]] * torch.as_tensor([-1, 1, -1, 1, -1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0, w, 0, w, 0])
    tgt["boxes"] = boxes
    tgt["theta"] = -target["theta"]
    tgt["polys"] = polys
    return flipped_image, tgt


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    tgt = target.copy()
    boxes = target["boxes"]
    scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
    polys = target["polys"]
    scaled_polys = polys * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height, ratio_width, ratio_height, ratio_width, ratio_height])
    tgt["boxes"] = scaled_boxes
    tgt["polys"] = scaled_polys

    h, w = size
    tgt["size"] = torch.tensor([h, w])

    return rescaled_image, tgt


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    tgt = target.copy()
    # should we do something wrt the original size?
    tgt["size"] = torch.tensor(padded_image.size[::-1])
    return padded_image, tgt


def rotate_poly_single(h, w, new_h, new_w, rotate_matrix_T, poly):
    poly[::2] = poly[::2] - (w - 1) * 0.5
    poly[1::2] = poly[1::2] - (h - 1) * 0.5
    coords = poly.reshape(4, 2)
    new_coords = np.matmul(coords, rotate_matrix_T) + np.array([(new_w - 1) * 0.5, (new_h - 1) * 0.5])
    rotated_polys = new_coords.reshape(-1, ).tolist()

    return rotated_polys

    # TODO: refactor the single - map to whole numpy computation
def rotate_poly(h, w, new_h, new_w, rotate_matrix_T, polys):
    rotate_poly_fn = partial(rotate_poly_single, h, w, new_h, new_w, rotate_matrix_T)
    rotated_polys = list(map(rotate_poly_fn, polys))
    return rotated_polys

class RotateAugmentation(object):
    def __init__(self,
                 # center=None,
                 scale=(0.7, 1.5),
                 border_value=0,
                 auto_bound=True,
                 rotate_range=(-180, 180),
                 rotate_ratio=1.0,
                 rotate_values=[0, 45, 90, 135, 180, 225, 270, 315],
                 rotate_mode='range',
                 small_filter=4):
        self.scale = random.uniform(scale[0], scale[1])
        self.border_value = border_value
        self.auto_bound = auto_bound
        self.rotate_range = rotate_range
        self.rotate_ratio = rotate_ratio
        self.rotate_values = rotate_values
        self.rotate_mode = rotate_mode
        self.small_filter = small_filter
        # self.center = center

    def __call__(self, img, target):
        # whether to rotate
        # 逆时针angle为负，顺时针为正
        if np.random.rand() > self.rotate_ratio:
            angle = 0.0
        else:
            if self.rotate_mode == 'range':
                angle = np.random.rand() * (self.rotate_range[1] - self.rotate_range[0]) + self.rotate_range[0]
            elif self.rotate_mode == 'value':
                random.shuffle(self.rotate_values)
                angle = self.rotate_values[0]
        img = np.array(img)
        # rotate image, copy from mmcv.imrotate
        h, w = img.shape[:2]
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(center, -angle, self.scale)
        matrix_T = copy.deepcopy(matrix[:2, :2]).T
        if self.auto_bound:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = h * sin + w * cos
            new_h = h * cos + w * sin
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))
        rotated_img = cv2.warpAffine(img, matrix, (w, h), borderValue=self.border_value)
        theta = target["theta"]
        oboxes = (torch.cat([box_xyxy_to_cxcywh(target["boxes"]), theta], dim=-1)).numpy()

        tgt = target.copy()
        if oboxes.shape[0] > 0:
            polys = target["polys"]
            rotated_polys = rotate_poly(img.shape[0], img.shape[1], h, w, matrix_T, polys)

            rotated_polys_np = np.array(rotated_polys)

            # add dimension in poly2mask
            rotated_boxes = bt.bbox2type(rotated_polys_np, 'obb')
            boxes, theta = rotated_boxes[:, :4], rotated_boxes[:, 4][..., None]
            boxes[:, :2] -= boxes[:, 2:] / 2.0
            boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
            tgt['polys'] = torch.as_tensor(rotated_polys_np)
            tgt['boxes'] = torch.as_tensor(boxes)
        tgt['theta'] = torch.as_tensor(theta)
        tgt['rotation'] = torch.tensor([angle, self.scale])
        tgt['size'] = torch.tensor([rotated_img.shape[0], rotated_img.shape[1]])
        return Image.fromarray(rotated_img), tgt

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        tgt = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = box_xyxy_to_cxcywh(target["boxes"])
            polys = target["polys"]
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            polys = polys / torch.tensor([w, h, w, h, w, h, w, h], dtype=torch.float32)
            tgt["boxes"] = boxes
            tgt["polys"] = polys
            tgt["img_size"] = torch.as_tensor([h, w])
        return image, tgt


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
    """

    def __init__(self, output_size=1333, aug_scale_min=0.3, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def rescale_target(self, scaled_size, image_size, target):
        # compute rescaled targets
        image_scale = torch.true_divide(scaled_size, image_size)
        ratio_height, ratio_width = image_scale

        tgt = target.copy()
        tgt["size"] = scaled_size

        boxes = target["boxes"]
        polys = target["polys"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        polys = polys * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height, ratio_width, ratio_height, ratio_width, ratio_height])
        tgt["boxes"] = scaled_boxes
        tgt["polys"] = polys

        return tgt

    def crop_target(self, region, target, thred=0.3):
        i, j, h, w = region
        fields = ["labels", "theta"]

        tgt = target.copy()
        tgt["size"] = torch.tensor([h, w])


        boxes, theta = target["boxes"], target["theta"]
        bbox1 = torch.cat([box_xyxy_to_cxcywh(boxes), theta], dim=1).numpy()
        bbox2 = np.array([[i + h * 0.5, j + w * 0.5, h, w, 0.0]])
        boxes_overlap = bt.bbox_overlaps(bbox1, bbox2).squeeze()
        keep = np.where(boxes_overlap >= thred)[0]
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_polys = target["polys"] - torch.as_tensor([j, i, j, i, j, i, j, i])
        tgt["boxes"] = cropped_boxes[keep, :]
        tgt["polys"] = cropped_polys[keep, :]

        for field in fields:
            tgt[field] = target[field][keep]
        return tgt

    def pad_target(self, padding, target):
        tgt = target.copy()
        return tgt

    def __call__(self, image, target=None):
        image_size = image.size
        image_size = torch.tensor(image_size[::-1])
        # 以desired_size为最长边，按比例确定out_desired_size,固定输出图片尺寸，out_desired_size根据样本尺寸不同也会不同
        out_desired_size = torch.true_divide(self.desired_size * image_size, max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        # 对图片进行缩放
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.min(torch.true_divide(scaled_size, image_size[0]), torch.true_divide(scaled_size, image_size[1]))
        scaled_size = (image_size * scale).round().int()

        scaled_image = F.resize(image, scaled_size.tolist())

        if target is not None:
            target = self.rescale_target(scaled_size, image_size, target)

        # randomly crop or pad images
        # 对缩放后尺寸>out_desired_size的进行裁剪
        if random_scale > 1:
            # Selects non-zero random offset (x, y) if scaled image is larger than desired_size.
            max_offset = scaled_size - out_desired_size
            offset = (max_offset * torch.rand(2)).floor().int()
            region = (offset[0].item(), offset[1].item(),
                      out_desired_size[0].item(), out_desired_size[1].item())
            output_image = F.crop(scaled_image, *region)
            if target is not None:
                target = self.crop_target(region, target)
        # 对缩放后尺寸＜out_desired_size的进行padding
        else:
            padding = out_desired_size - scaled_size
            output_image = F.pad(scaled_image, (0, 0, padding[1].item(), padding[0].item()))
            if target is not None:
                target = self.pad_target(padding, target)

        return output_image, target


class RandomDistortion(object):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0.5):
        self.prob = prob
        self.tfm = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, target=None):
        if np.random.random() < self.prob:
            return self.tfm(img), target
        else:
            return img, target

