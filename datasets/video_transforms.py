import math
import copy
import torch
import random
import numpy as np
import copy
import PIL
import numbers
import cv2
from PIL import Image
from util.box_ops import interpolate, box_xyxy_to_cxcywh


# ==================
# torch_videovision
# ==================
def convert_img(img):
    """Converts (H, W, C) numpy.ndarray to (C, W, H) format"""
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)
    return img


class ClipToTensor(object):
    """Convert a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    to a torch.FloatTensor of shape (C x m x H x W) in the range [0, 1.0]
    """

    def __init__(self, channel_nb=3, div_255=True, numpy=False):
        self.channel_nb = channel_nb
        self.div_255 = div_255
        self.numpy = numpy

    def __call__(self, clip):
        """
        Args: clip (list of numpy.ndarray): clip (list of images)
        to be converted to tensor.
        """
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            assert ch == self.channel_nb, "Got {0} instead of 3 channels".format(ch)
        elif isinstance(clip[0], Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image\
            but got list of {0}".format(
                    type(clip[0])
                )
            )

        np_clip = np.zeros([self.channel_nb, len(clip), int(h), int(w)])

        # Convert
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError(
                    "Expected numpy.ndarray or PIL.Image\
                but got list of {0}".format(
                        type(clip[0])
                    )
                )
            img = convert_img(img)
            np_clip[:, img_idx, :, :] = img
        if self.numpy:
            if self.div_255:
                np_clip = np_clip / 255
            return np_clip

        else:
            tensor_clip = torch.from_numpy(np_clip)

            if not isinstance(tensor_clip, torch.FloatTensor):
                tensor_clip = tensor_clip.float()
            if self.div_255:
                tensor_clip = tensor_clip.div(255)
            return tensor_clip


def _is_tensor_clip(clip):
    return torch.is_tensor(clip) and clip.ndimension() == 4


def normalize(clip, mean, std, inplace=False):
    if not _is_tensor_clip(clip):
        raise TypeError("tensor is not a torch clip.")

    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

    return clip


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def resize_clip(clip, size, interpolation="bilinear"):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == "bilinear":
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = [cv2.resize(img, size, interpolation=np_inter) for img in clip]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == "bilinear":
            pil_inter = PIL.Image.NEAREST
        else:
            pil_inter = PIL.Image.BILINEAR
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError(
            "Expected numpy.ndarray or PIL.Image"
            + "but got list of {0}".format(type(clip[0]))
        )
    return scaled


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h : min_h + h, min_w : min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip]
    else:
        raise TypeError(
            "Expected numpy.ndarray or PIL.Image"
            + "but got list of {0}".format(type(clip[0]))
        )
    return cropped


# =================
# Video Transforms
# =================
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, video, targets):
        for t in self.transforms:
            video, targets = t(video, targets)
        return video, targets

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __init__(self, channel_nb=3, div_255=True, numpy=False):
        self.ClipToTensor = ClipToTensor(channel_nb, div_255, numpy)

    def __call__(self, video, targets):
        return self.ClipToTensor(video), targets


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, video, targets):
        
        video = normalize(video, mean=self.mean, std=self.std)  # torch functional videotransforms
        if targets is None:

            return video, None
        targets = targets.copy()
        h, w = video.shape[-2:]
        #if "boxes" in targets[0]:  
        for box_k in ["sboxes", "oboxes"]: # apply for every image of the clip
            for i_tgt in range(len(targets)):
                boxes = targets[i_tgt][box_k]
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                targets[i_tgt][box_k] = boxes
        return video, targets


def hflip(clip, targets):
    if isinstance(clip[0], np.ndarray):
        flipped_clip = [np.fliplr(img) for img in clip]  # apply for every image of the clip
        h, w = clip[0].shape[:2]
    elif isinstance(clip[0], PIL.Image.Image):
        flipped_clip = [img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip]  # apply for every image of the clip
        w, h = clip[0].size

    targets = targets.copy()
    for box_k in ["sboxes", "oboxes"]:  # apply for every image of the clip
        for i_tgt in range(len(targets)):
            boxes = targets[i_tgt][box_k]
            boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor(
                [-1, 1, -1, 1]
            ) + torch.as_tensor([w, 0, w, 0])
            targets[i_tgt][box_k] = boxes
    
    return flipped_clip, targets


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, video, targets):
        if random.random() < self.p:
            return hflip(video, targets)
        return video, targets


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, video, targets):
        if random.random() < self.p:
            return self.transforms1(video, targets)
        return self.transforms2(video, targets)


def resize(clip, targets, size, max_size=None):
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
    
    if isinstance(clip[0], PIL.Image.Image):
        s = clip[0].size
    elif isinstance(clip[0], np.ndarray):
        h, w, ch = list(clip[0].shape)
        s = [w, h]
    else:
        raise NotImplementedError
    
    size = get_size(
        s, size, max_size
    )  # apply for first image, all images of the same clip have the same h w
    
    rescaled_clip = resize_clip(clip, size)  # torch video transforms functional
    
    if isinstance(clip[0], np.ndarray):
        h2, w2, c2 = list(rescaled_clip[0].shape)
        s2 = [w2, h2]
    elif isinstance(clip[0], PIL.Image.Image):
        s2 = rescaled_clip[0].size
    else:
        raise NotImplementedError

    if targets is None:
        return rescaled_clip, None
    
    ratios = tuple(float(s_mod) / float(s_orig) for s_mod, s_orig in zip(s2, s))
    ratio_width, ratio_height = ratios
    
    targets = targets.copy()
    
    for box_k in ["sboxes", "oboxes"]:
        for i_tgt in range(len(targets)):  # apply for every image of the clip
            boxes = targets[i_tgt][box_k]
            scaled_boxes = boxes * torch.as_tensor(
                [ratio_width, ratio_height, ratio_width, ratio_height]
            )
            targets[i_tgt][box_k] = scaled_boxes
    
    for area_k in ["sarea", "oarea"]: # TODO: not sure if it is needed to do for all images from the clip
        for i_tgt in range(len(targets)):  # apply for every image of the clip
            area = targets[i_tgt][area_k]
            scaled_area = area * (ratio_width * ratio_height)
            targets[i_tgt][area_k] = scaled_area
    
    h, w = size
    for i_tgt in range(len(targets)):  # TODO: not sure if it is needed to do for all images from the clip
        targets[i_tgt]["size"] = torch.tensor([h, w])

    return rescaled_clip, targets


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, video, target=None):
        size = random.choice(self.sizes)
        return resize(video, target, size, self.max_size)


def crop(clip, orig_targets, region, target_only=False):
    if not target_only:
        cropped_clip = crop_clip(clip, *region)
        # cropped_clip = [F.crop(img, *region) for img in clip] # other possibility is to use torch_videovision.torchvideotransforms.functional.crop_clip

    targets = copy.deepcopy(orig_targets)
    i, j, h, w = region
    
    # should we do something wrt the original size?
    for i_tgt in range(len(targets)):  # TODO: not sure if it is needed to do for all images from the clip
        targets[i_tgt]["size"] = torch.tensor([h, w])

    fields = ['sarea', 'oarea', 'so_traj_ids', 'sclss', 'oclss', 'vclss', 'raw_vclss', 
              'orig_size', 'size', 'num_svo', 'svo_ids']
    
    for so_k in ["s", "o"]:
        if so_k+"boxes" in targets[0]:
            for i_tgt in range(len(targets)):  # apply for every image of the clip
                boxes = targets[i_tgt][so_k+"boxes"]
                max_size = torch.as_tensor([w, h], dtype=torch.float32)
                cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
                cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
                cropped_boxes = cropped_boxes.clamp(min=0)
                area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
                targets[i_tgt][so_k+"boxes"] = cropped_boxes.reshape(-1, 4)
                targets[i_tgt][so_k+"area"] = area
            fields.append(so_k+"boxes")
    
    # remove elements for which the boxes or masks that have zero area
    # favor boxes selection when defining which elements to keep
    # this is compatible with previous implementation
    
    for i_tgt in range(len(targets)):
        cropped_boxes = targets[i_tgt]["sboxes"].reshape(-1, 2, 2)
        s_keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        cropped_boxes = targets[i_tgt]["oboxes"].reshape(-1, 2, 2)
        o_keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        
        keep = s_keep*o_keep
      
        for field in fields:
            if field in targets[i_tgt]:   
                if field in ["orig_size", "size"]:
                    continue
                elif field == "raw_vclss":
                    targets[i_tgt][field] = [targets[i_tgt][field][i] for i in range(len(keep)) if keep[i]==True]
                elif field == "num_svo":
                    targets[i_tgt][field] = keep.sum()
                else:
                    targets[i_tgt][field] = targets[i_tgt][field][keep]
    if not target_only:
        return cropped_clip, targets
    else:
        return targets


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, scale: list, respect_boxes: bool = False, by_ratio = True):
        self.min_size = min_size
        self.max_size = max_size
        self.scale = scale
        self.by_ratio = by_ratio
        self.respect_boxes = respect_boxes  # if True we can't crop a box out

    def resize_by_ratio(self, clip, targets, init_sboxes, init_oboxes, 
                        img_height, img_width, scale, ratio=(3.0 / 4.0, 4.0 / 3.0)):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        orig_targets = copy.deepcopy(targets)

        height, width = img_height, img_width
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))

        for _ in range(20):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            i = torch.randint(0, max(1,height - h + 1), size=(1,)).item()
            j = torch.randint(0, max(1,width - w + 1), size=(1,)).item()
            region = i, j, h, w
            result_targets = crop(clip, targets, region, target_only=True)
           
            sbox_sum = sum(len(result_targets[i_patience]["sboxes"]) for i_patience in range(len(result_targets)))
            obox_sum = sum(len(result_targets[i_patience]["oboxes"]) for i_patience in range(len(result_targets)))
            if (sbox_sum==init_sboxes) and (obox_sum==init_oboxes): # make sure 
                result_clip, result_targets = crop(clip, targets, region)
                return result_clip, result_targets
        
        return clip, orig_targets

    def __call__(self, clip, targets: dict):
        orig_targets = copy.deepcopy(targets)  # used to conserve ALL BOXES ANYWAY
        init_sboxes = sum(len(targets[i_tgt]["sboxes"]) for i_tgt in range(len(targets)))
        init_oboxes = sum(len(targets[i_tgt]["oboxes"]) for i_tgt in range(len(targets)))
        max_patience = 100  # TODO: maybe it is gonna requery lots of time with a clip than an image as it involves more boxes
        
        if isinstance(clip[0], PIL.Image.Image):
            h = clip[0].height
            w = clip[0].width
        elif isinstance(clip[0], np.ndarray):
            h = clip[0].shape[0]
            w = clip[0].shape[1]
        else:
            raise NotImplementedError
        
        if self.by_ratio:
            result_clip, result_targets = self.resize_by_ratio(clip, targets, init_sboxes, init_oboxes, h, w, self.scale) 
            return result_clip, result_targets
        
        for _ in range(max_patience):
            tw = random.randint(self.min_size, min(w, self.max_size))
            th = random.randint(self.min_size, min(h, self.max_size))
        
            ##region = T.RandomCrop.get_params(clip[0], [th, tw]) # h w sizes are the same for all images of the clip; we can just get parameters for the first image
            
            if h + 1 < th or w + 1 < tw:
                raise ValueError("Required crop size {} is larger then input image size {}".format(
                        (th, tw), (h, w)))

            if w == tw and h == th:
                region = 0, 0, h, w
            else:
                i = torch.randint(0, h - th + 1, size=(1,)).item()
                j = torch.randint(0, w - tw + 1, size=(1,)).item()
                region = i, j, th, tw
            
            #result_clip, result_targets = crop(clip, targets, region)
            result_targets = crop(clip, targets, region, target_only=True)  # to speed up sbox/obox_sum calculation, only crop target
            sbox_sum = sum(len(result_targets[i_patience]["sboxes"])for i_patience in range(len(result_targets)))
            obox_sum = sum(len(result_targets[i_patience]["oboxes"])for i_patience in range(len(result_targets)))
            
            if (not self.respect_boxes) or ((sbox_sum==init_sboxes) and (obox_sum==init_oboxes)):
                result_clip, result_targets = crop(clip, targets, region)
                return result_clip, result_targets
        
        return clip, orig_targets