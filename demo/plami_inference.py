import torch
from plami.utils import disable_torch_init
from transformers import AutoTokenizer, CLIPImageProcessor
from plami.model.language_model.plami_llama import PLAMiLlamaForCausalLM
from plami.mm_utils import tokenizer_image_token
from plami.conversation import conv_templates, SeparatorStyle
from plami.constants import IMAGE_TOKEN_INDEX
from plami.train.train import DataArguments

from functools import partial
import os
import numpy as np
import cv2

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True


def show_mask(mask, image, random_color=True, img_trans=0.9, mask_trans=0.5, return_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3) * 255], axis=0)
    else:
        color = np.array([30, 144, 255])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    image = cv2.addWeighted(image, img_trans, mask_image.astype('uint8'), mask_trans, 0)
    if return_color:
        return image, mask_image
    else:
        return image


class PLAMi():
    def __init__(self, model_path, device='cuda'):
        disable_torch_init()
        model_path = os.path.expanduser(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=2048,
            padding_side="right",
            use_fast=True
        )
        self.model = PLAMiLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.tokenizer.pad_token = self.tokenizer.unk_token

        # 指定视觉塔的名称
        vision_tower_name = 'openai/clip-vit-large-patch14-336'

        # 从预定义模型中创建实例
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)

        spi_tokens = ['<mask>', '<pos>']
        self.tokenizer.add_tokens(spi_tokens, special_tokens=True)

        for m in self.model.modules():
            m.tokenizer = self.tokenizer

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=torch.float16, device=device)

        begin_str = """<image>\n\nThis provides an overview of the medical picture.\n"""

        short_question = 'Please give me the name of <mask><pos> in the medical image. Using only one word or phrase.'

        conv = conv_templates['plami_v1'].copy()
        qs = begin_str + short_question
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        self.input_ids_short = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                                     return_tensors='pt').unsqueeze(0).to(self.model.device)

        detailed_question = 'Could you describe the region shown as <mask><pos> in the medical image in great detail, please?'

        conv = conv_templates['plami_v1'].copy()
        qs = begin_str + detailed_question
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        self.input_ids_detailed = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                                        return_tensors='pt').unsqueeze(0).to(self.model.device)

        self.stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    def plami_predict(self, img, mask, type=None):
        image = self.image_processor.preprocess(img,
                                                do_center_crop=False,
                                                return_tensors='pt')['pixel_values'][0]

        image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                size=(336, 336),
                                                mode='bilinear',
                                                align_corners=False).squeeze(0)

        masks = torch.Tensor(mask).unsqueeze(0).to(self.model.device)

        if type == 'short description':
            input_ids = self.input_ids_short
        else:
            input_ids = self.input_ids_detailed

        # self.model.model.tokenizer = self.tokenizer

        with torch.inference_mode():

            self.model.orig_forward = self.model.forward
            self.model.forward = partial(self.model.orig_forward,
                                         img_metas=[None],
                                         masks=[masks.half()])

            output_ids = self.model.generate(
                input_ids,
                images=image.unsqueeze(0).half().to(self.model.device),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                num_beams=1,
                # stopping_criteria=[stopping_criteria]
            )

            self.model.forward = self.model.orig_forward

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
                input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:],
                                              skip_special_tokens=True)[0]

        outputs = outputs.strip()
        if outputs.endswith(self.stop_str):
            outputs = outputs[:-len(self.stop_str)]
        outputs = outputs.strip()
        if ':' in outputs:
            outputs = outputs.split(':')[1]

        outputs_list = outputs.split('.')
        outputs_list_final = []
        outputs_str = ''
        for output in outputs_list:
            if output not in outputs_list_final:
                if output == '':
                    continue
                outputs_list_final.append(output)
                outputs_str += output + '.'
            else:
                break
        return outputs_str
