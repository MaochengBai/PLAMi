import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = -2
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True


    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    import torch
    import torch.nn.functional as F

    @torch.no_grad()
    def forward(self, images):

        if type(images) is list:
            image_features = []  # 存储选定层（如 res4 或 res6）的特征
            image_features_dict = []  # 存储所有层的特征字典
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      output_hidden_states=True)
                raw_features = self.feature_select(image_forward_out).to(image.dtype)

                # 转置并重塑特征，适用于插值
                reshaped_features = raw_features.permute(0, 2, 1).reshape(raw_features.shape[0], 1024, 24, 24)
                reshaped_features = reshaped_features.float()
                # 构建特征字典并包括所有层
                image_features_dict = {
                    'stem': F.interpolate(reshaped_features, size=(128, 128), mode='bilinear', align_corners=False),
                    'res2': F.interpolate(reshaped_features, size=(128, 128), mode='bilinear', align_corners=False),
                    'res3': F.interpolate(reshaped_features, size=(64, 64), mode='bilinear', align_corners=False),
                    'res4': F.interpolate(reshaped_features, size=(32, 32), mode='bilinear', align_corners=False),
                    'res5': F.interpolate(reshaped_features, size=(16, 16), mode='bilinear', align_corners=False),
                    'clip_vis_dense': F.interpolate(reshaped_features, size=(16, 16), mode='bilinear',
                                                    align_corners=False),
                    'res6': raw_features  # 使用未变形的原始特征
                }
                image_features_dict.append(image_features_dict)
                image_features.append(feature_dict['res6'])  # 这里使用res6作为返回的特征示例
        else:
            # 处理单个图像
            image_forward_out = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                  output_hidden_states=True)
            raw_features = self.feature_select(image_forward_out).to(images.dtype)

            # 同样进行转置和重塑
            reshaped_features = raw_features.permute(0, 2, 1).reshape(raw_features.shape[0], 1024, 24, 24)
            reshaped_features = reshaped_features.float()
            # 创建特征字典并包括所有层
            image_features_dict = {
                'stem': F.interpolate(reshaped_features, size=(128, 128), mode='bilinear', align_corners=False),
                'res2': F.interpolate(reshaped_features, size=(128, 128), mode='bilinear', align_corners=False),
                'res3': F.interpolate(reshaped_features, size=(64, 64), mode='bilinear', align_corners=False),
                'res4': F.interpolate(reshaped_features, size=(32, 32), mode='bilinear', align_corners=False),
                'res5': F.interpolate(reshaped_features, size=(16, 16), mode='bilinear', align_corners=False),
                'clip_vis_dense': F.interpolate(reshaped_features, size=(16, 16), mode='bilinear', align_corners=False),
                'res6': raw_features  # 使用未变形的原始特征
            }
            image_features = image_features_dict['res6']

        return image_features, image_features_dict

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



