import time

import megengine as mge
import numpy as np
import torch

from convert_weights import convert
from models.senet import *
from models.torch_models import *

mge_model = senet154(True)
# mge_model = seresnet50(True)

# print("\ndownload manually if speed is too slow, or just load without pretrained weights, and convert state dict from torch to megneinge\n")
# torch_model = torch_senet154(True)
# torch_model.load_state_dict(torch.load(
#     './senet154-c7b49a05.pth', map_location='cpu'))
# try:
#     torch_model = torch_seresnet50(True)
#     # torch_model.load_state_dict(torch.load('./se_resnet50-ce0d4300.pth', map_location='cpu'))
# except:
#     print("\nfail to download, now load model without pretrained weights, and convert state dict from torch to megneinge\n")
#     torch_model = torch_seresnet50()
#     torch_state_dict =  torch_model.state_dict()
#     new_dit = convert(torch_model, torch_state_dict)
#     mge_model.load_state_dict(new_dit)
# torch_model = torch_seresnet50()
torch_model = torch_senet154()
torch_state_dict = torch_model.state_dict()
new_dit = convert(torch_model, torch_state_dict)
mge_model.load_state_dict(new_dit)

mge_model.eval()
torch_model.eval()

torch_time = meg_time = 0.0


def test_func(mge_out, torch_out):
    result = np.isclose(mge_out, torch_out, rtol=1e-3)
    ratio = np.mean(result)
    allclose = np.all(result) > 0
    abs_err = np.mean(np.abs(mge_out - torch_out))
    std_err = np.std(np.abs(mge_out - torch_out))
    return ratio, allclose, abs_err, std_err


def softmax(logits):
    logits = logits - logits.max(-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(-1, keepdims=True)


for i in range(15):
    results = []
    inp = np.random.randn(2, 3, 224, 224)
    mge_inp = mge.tensor(inp, dtype=np.float32)
    torch_inp = torch.tensor(inp, dtype=torch.float32)

    if torch.cuda.is_available():
        torch_inp = torch_inp.cuda()
        torch_model.cuda()

    st = time.time()
    mge_out = mge_model(mge_inp)
    meg_time += time.time() - st

    st = time.time()
    torch_out = torch_model(torch_inp)
    torch_time += time.time() - st

    if torch.cuda.is_available():
        torch_out = torch_out.detach().cpu().numpy()
    else:
        torch_out = torch_out.detach().numpy()
    mge_out = mge_out.numpy()
    mge_out = softmax(mge_out)
    torch_out = softmax(torch_out)
    ratio, allclose, abs_err, std_err = test_func(mge_out, torch_out)
    results.append(allclose)
    print(f"Result: {allclose}, {ratio*100 : .4f}% elements is close enough\n which absolute error is  {abs_err} and absolute std is {std_err}")

assert all(results), "not aligned"

print(f"meg time: {meg_time}, torch time: {torch_time}")
