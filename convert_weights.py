import argparse
import os

import megengine as mge
import numpy as np
import torch
import torch.nn as nn

from models.senet import (senet154, seresnet18, seresnet34, seresnet50,
                          seresnet101, seresnet152, seresnext26_32x4d,
                          seresnext50_32x4d, seresnext101_32x4d)
from models.torch_models import (torch_senet154, torch_seresnet18,
                                 torch_seresnet34, torch_seresnet50,
                                 torch_seresnet101, torch_seresnet152,
                                 torch_seresnext26_32x4d,
                                 torch_seresnext50_32x4d,
                                 torch_seresnext101_32x4d)


def get_atttr_by_name(torch_module, k):
    name_list = k.split('.')
    sub_module = getattr(torch_module, name_list[0])
    if len(name_list) != 1:
        for i in name_list[1:-1]:
            try:
                sub_module = getattr(sub_module, i)
            except:
                sub_module = sub_module[int(i)]
    return sub_module


def convert(torch_model, torch_dict):
    new_dict = {}
    for k, v in torch_dict.items():
        data = v.numpy()
        sub_module = get_atttr_by_name(torch_model, k)
        is_conv = isinstance(sub_module, nn.Conv2d)
        if is_conv:
            groups = sub_module.groups
            is_group = groups > 1
        else:
            is_group = False
        if "weight" in k and is_group:
            out_ch, in_ch, h, w = data.shape
            data = data.reshape(groups, out_ch // groups, in_ch, h, w)
        if "bias" in k:
            if is_conv:
                data = data.reshape(1, -1, 1, 1)
        if "num_batches_tracked" in k:
            continue
        new_dict[k.replace("se_module", "se_block")] = data
    return new_dict


def main(torch_name):
    try:
        torch_model = eval("torch_" + torch_name)(pretrained=True)
    except:
        print("If you can't download the pretrained weights, just download it from timm repo in github")
        torch_model = eval("torch_" + torch_name)()
        torch_model.load_state_dict(torch.load(
            './senet154-c7b49a05.pth', map_location='cpu'))
    torch_state_dict = torch_model.state_dict()
    model = eval(torch_name)()

    new_dict = convert(torch_model, torch_state_dict)

    model.load_state_dict(new_dict)
    os.makedirs('pretrained', exist_ok=True)
    mge.save(new_dict, os.path.join('pretrained', torch_name + '.pkl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default='senet154',
        help=f"which model to convert from torch to megengine",
    )
    args = parser.parse_args()
    main(args.model)
