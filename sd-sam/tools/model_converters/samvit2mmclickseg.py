# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import math

from focsam.backbones.sam_vit_ada import Adapter_SAMWindowViT
import mmengine
import torch
import torch.nn as nn


def convert_vit(ckpt , ckpt_2):

    new_ckpt = OrderedDict()



    for k, v in ckpt.items():

        if k.startswith('image_encoder'):
            new_k = k.replace('image_encoder', 'backbone')
            # if 'attn' in new_k:
            #     new_k = new_k.replace('attn', 'attn.attn')
            # elif 'mlp' in new_k:
            #     new_k = new_k.replace('mlp', 'mlp.mlp')

        elif k.startswith('prompt_encoder'):
            new_k = k.replace('prompt_encoder', 'neck')
            if 'pe_layer' in k:
                new_k = new_k.replace('pe_layer', 'pos_embed_layer')
            if 'point_embeddings' in k:
                new_k = new_k.replace('point_embeddings', 'point_embeds')
        elif k.startswith('mask_decoder'):
            new_k = k.replace('mask_decoder', 'decode_head')


        new_ckpt[new_k] = v

    for k, v in ckpt_2.items():
        ori_k = k
        new_k = k.replace(ori_k, 'backbone.' + ori_k)
        new_ckpt[new_k] = v

    return new_ckpt

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()

def add_missing_keys(state_dict, model):
    """
    在原权重字典中添加缺失的键值对，使其匹配模型。

    :param state_dict: 原权重字典
    :param model: 现在的模型
    :return: 更新后的权重字典
    """
    new_state_dict = state_dict.copy()
    for name, param in model.named_parameters():
        if name not in new_state_dict:
            # 处理 attn 模块的 down_fn 和 up_fn
            # if "attn.down_fn" in name or "attn.up_fn" in name:
            #     module_name = name.rsplit('.', 2)[0]
            #     module = model.get_submodule(module_name)
            #     if "down_fn" in name:
            #         new_param = nn.Linear(1280, 64, bias=True).weight  # 假设中间维度为 64
            #     else:
            #         new_param = nn.Linear(64, 1280, bias=True).weight
            #
            #     name = name.replace('blocks', 'backbone.blocks')
            #     new_state_dict[name] = new_param
            # # 处理 mlp 模块的 down_fn 和 up_fn
            # elif "mlp.down_fn" in name or "mlp.up_fn" in name:
            #     module_name = name.rsplit('.', 2)[0]
            #     module = model.get_submodule(module_name)
            #     if "down_fn" in name:
            #         new_param = nn.Linear(1280, 64, bias=True).weight  # 假设中间维度为 64
            #     else:
            #         new_param = nn.Linear(64, 1280, bias=True).weight
            #
            #     name = name.replace('blocks', 'backbone.blocks')
            #     new_state_dict[name] = new_param

            if "attn.down_fn" in name or "attn.up_fn" in name:
                module_name = name.rsplit('.', 2)[0]
                module = model.get_submodule(module_name)
                if "down_fn" in name:
                    linear_layer = nn.Linear(768, 32, bias=True)
                else:
                    linear_layer = nn.Linear(32, 768, bias=True)
                # 使用 _init_weights 进行初始化
                _init_weights(linear_layer)
                name = name.replace('blocks', 'backbone.blocks')
                if 'weight' in name :
                    new_state_dict[name] = linear_layer.weight
                if 'bias' in name:
                    new_state_dict[name] = linear_layer.bias
                # 处理 mlp 模块的 down_fn 和 up_fn
            elif "mlp.down_fn" in name or "mlp.up_fn" in name:
                module_name = name.rsplit('.', 2)[0]
                module = model.get_submodule(module_name)
                if "down_fn" in name:
                    linear_layer = nn.Linear(768, 32, bias=True)
                else:
                    linear_layer = nn.Linear(32, 768, bias=True)
                # 使用 _init_weights 进行初始化
                _init_weights(linear_layer)
                name = name.replace('blocks', 'backbone.blocks')
                if 'weight' in name:
                    new_state_dict[name] = linear_layer.weight
                if 'bias' in name:
                    new_state_dict[name] = linear_layer.bias

    return new_state_dict


# 假设 model 是你的现在的模型实例




def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
        'MMSegmentation style.')
    parser.add_argument('--src', help='src model path or url')
    parser.add_argument('--dst', help='save path')
    args = parser.parse_args()

    checkpoint = torch.load(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # deit checkpoint
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    # weight = convert_vit(state_dict)
    weight = state_dict
    mmengine.mkdir_or_exist(osp.dirname(args.dst))

    # model = Adapter_SAMWindowViT(depth=12)
    # 假设 state_dict 是你加载的原始权重字典
    # state_dict = torch.load('/media/data3/qh/project/focsam-master/pretrain/sam_pretrain_vit_base_ada.pth',
    #                         map_location='cpu')

    # new_state_dict = add_missing_keys(state_dict, model)

    state_dict = torch.load('/media/data3/qh/project/focsam-master/pretrain/mae_pretrain_vit_base.pth',
                            map_location='cpu')
    print(state_dict['model'].keys(),'sgaikfiwefjioluwoiasjfoiaioesjgfoasfiofuoqwjofla\n',weight.keys())
    torch.save(convert_vit( weight ,state_dict['model']), '/media/data3/qh/project/focsam-master/pretrain/mae_pretrain_vit_base_sam_other_2.pth')

    # torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
