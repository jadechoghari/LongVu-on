# pyre-unsafe
import copy
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, Dinov2Config, Dinov2Model, SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel
from abc import ABC, abstractmethod
import torch.nn as nn


class ProcessorWrapper:
    def __init__(
        self,
        transform,
        height=378,
        width=378,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
    ):
        self._crop_size = {
            "height": height,
            "width": width,
        }
        self._transforms = transform
        # print(transform)
        self.image_mean = image_mean

    @property
    def crop_size(self):
        return self._crop_size

    def preprocess(self, image, return_tensors="pt"):
        # Ensure image is a PIL Image
        output = {}
        output["pixel_values"] = [self._transforms(image)]
        return output


class BaseVisionTower(nn.Module):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.args = args

        self.vision_tower_name = vision_tower_name
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self.unfreeze_mm_vision_tower = getattr(args, "unfreeze_mm_vision_tower", False)
        self.delay_load = delay_load

    @abstractmethod
    def load_model(self, device_map=None):
        raise NotImplementedError("Subclasses must implement load_model")

    @abstractmethod
    def _forward(self, images):
        raise NotImplementedError("Subclasses must implement forward")

    def forward(self, images):
        if type(images) is list:
            image_features = [self._forward(image.unsqueeze(0)) for image in images]
        else:
            image_features = self._forward(images)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        # Dynamically infer the dtype from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower, "dtype"):
            return self.vision_tower.dtype
        else:
            params = list(self.vision_tower.parameters())
            return (
                params[0].dtype if len(params) > 0 else torch.float32
            )  # Default to torch.float32 if no parameters

    @property
    def device(self):
        # Dynamically infer the device from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower, "device"):
            return self.vision_tower.device
        else:
            params = list(self.vision_tower.parameters())
            return (
                params[0].device if len(params) > 0 else torch.device("cpu")
            )  # Default to CPU if no parameters

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        try:
            return self.config.hidden_size
        except:
            return self._hidden_size

    @property
    def image_size(self):  # resolution
        # return self.config.image_size
        try:
            return self.config.image_size
        except:
            return self._image_size

    @property
    def patch_size(self):
        # return self.config.patch_size
        try:
            return self.config.patch_size
        except:
            return self._patch_size

    @property
    def num_patches_per_side(self):
        if self._interp_size is not None:
            return int(self._interp_size**0.5)
        try:
            return self.image_size // self.patch_size
        except:
            return self._num_patches_per_side

    @property
    def num_patches(self):
        if self._interp_size is not None:
            return self._interp_size
        try:
            return self.num_patches_per_side**2
        except:
            return self._num_patches


class DinoVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super(DinoVisionTower, self).__init__(vision_tower, args, delay_load)

        model_path = "facebook/dinov2-giant"
        base_model_name, res, interp = model_path, 378, 576
        self._vision_tower_name = vision_tower
        self.vision_tower_name = base_model_name
        self._image_size = res
        self._interp_size = interp
        self._patch_size = 14  # default patch size

        if not self.delay_load:
            self.load_model()
        else:
            self.cfg_only = Dinov2Config.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):

        self.vision_tower = Dinov2Model.from_pretrained(self.vision_tower_name)
        """ValueError: Dinov2Model does not support `device_map='auto'`. To implement support, the model class needs to implement the `_no_split_modules` attribute."""
        self.vision_tower._no_split_modules = ["Dinov2SwiGLUFFN"]

        _image_size = self.vision_tower.config.image_size
        if self._image_size is None:
            self._image_size = _image_size

        # increase shortest edge to prevent edge case crops
        default_shortest_ratio = 8 / 7  # 224/256
        # shortest_edge = int(default_shortest_ratio * self._image_size)
        shortest_edge = self._image_size

        processor = AutoImageProcessor.from_pretrained(
            self.vision_tower_name,
            crop_size=dict(height=self._image_size, width=self._image_size),
            size=dict(shortest_edge=shortest_edge),
        )
        self.image_processor = processor

        # Assign the output channels of the projection convolution as the hidden size
        self._hidden_size = (
            self.vision_tower.embeddings.patch_embeddings.projection.out_channels
        )
        # Assign the first value of the stride of the projection convolution as the patch size
        self._patch_size = (
            self.vision_tower.embeddings.patch_embeddings.projection.stride[0]
        )

        # print(self._hidden_size, self._patch_size)

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    @property
    def image_size(self):
        return self._image_size

    def feature_select(self, outputs):
        sequence_output = outputs[
            "last_hidden_state"
        ]  # batch_size, sequence_length, hidden_size

        if self.select_feature == "cls_patch":
            image_features = sequence_output
        elif self.select_feature == "patch":
            image_features = sequence_output[:, 1:]
        elif self.select_feature == "cls":
            image_features = sequence_output[:, 0]
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size**0.5)
            h = w = int(num_tokens**0.5)

            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).to(image_features.dtype)

            # Permute the dimensions back to (b, target_h, target_w, dim)
            image_features = image_features.permute(0, 2, 3, 1).contiguous()

            # Flatten the spatial dimensions (target_h, target_w) into a single dimension
            image_features = image_features.flatten(1, 2)

        return image_features

    def _forward(self, images):
        # logger.warning(f"images shape: {images.shape}")
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_forward_outs = self.vision_tower.forward(
                images.to(device=self.device, dtype=self.dtype)
            )
            # logger.warning(f"image_forward_outs shape: {image_forward_outs['last_hidden_state'].shape}")
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            # logger.warning(f"image_features shape: {image_features.shape}")
            interp_features = self.interpolate(image_features)
            # logger.warning(f"interp_features shape: {interp_features.shape}")
            return interp_features

    @property
    def num_patches_per_side(self):
        return int(self.num_patches**0.5)

    @property
    def num_patches(self):
        if self._interp_size is None:
            return (self._image_size // self._patch_size) ** 2
        else:
            return self._interp_size
        
        
# from .siglip_encoder import SiglipVisionTower
class SiglipVisionTower(BaseVisionTower):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super(SiglipVisionTower, self).__init__(vision_tower_name, args, delay_load)
        
        model_path = "google/siglip-so400m-patch14-384"
        base_model_name, res, interp = model_path, 384, 576
        self.vision_tower_name = base_model_name
        self._image_size = res if res is not None else 512
        self._interp_size = interp
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self._hidden_size = 1152

    def load_model(self, device_map=None):
        self.vision_model = "siglip"
        # clip_model, processor = create_model_from_pretrained(self.vision_tower_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)

        # self.vision_tower = clip_model.visual.trunk
        self.vision_tower.output_tokens = True

        self._hidden_size = self.vision_tower.config.hidden_size
        self._image_size = self.vision_tower.config.image_size
        self._patch_size = self.vision_tower.config.patch_size
        self.image_processor = SiglipImageProcessor.from_pretrained(
            self.vision_tower_name
        )

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size**0.5)
            h = w = int(num_tokens**0.5)

            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).to(image_features.dtype)

            # Permute the dimensions back to (b, target_h, target_w, dim)
            image_features = image_features.permute(0, 2, 3, 1).contiguous()

            # Flatten the spatial dimensions (target_h, target_w) into a single dimension
            image_features = image_features.flatten(1, 2)

        return image_features

    def _forward(self, images, interpolate_token=576):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_features = self.vision_tower.forward(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            ).hidden_states[-1]
            interp_features = self.interpolate(image_features)
            return interp_features


def build_vision_tower_aux_list(vision_tower_cfg, **kwargs):
    vision_tower_aux_name_list = getattr(
        vision_tower_cfg,
        "mm_vision_tower_aux_list",
        getattr(vision_tower_cfg, "vision_tower_aux_list", None),
    )
    vision_tower_aux_token_len_list = getattr(
        vision_tower_cfg,
        "mm_vision_tower_aux_token_len_list",
        getattr(vision_tower_cfg, "vision_tower_aux_token_len_list", None),
    )
    vision_tower_aux_list = []
    for vision_tower_aux_name, vision_tower_aux_token_len in zip(
        vision_tower_aux_name_list, vision_tower_aux_token_len_list
    ):
        config = copy.deepcopy(vision_tower_cfg)
        vision_tower_aux_name += "-interp{}".format(vision_tower_aux_token_len)
        if "siglip" in vision_tower_aux_name.lower():
            vision_tower_aux_list.append(
                SiglipVisionTower(vision_tower_aux_name, args=config, **kwargs)
            )

        # SSL-based Vision Towers
        elif "dinov2" in vision_tower_aux_name.lower():
            vision_tower_aux_list.append(
                DinoVisionTower(vision_tower_aux_name, args=config, **kwargs)
            )
        else:
            raise ValueError(f"Unknown vision tower: {vision_tower_aux_name}")
    return vision_tower_aux_list