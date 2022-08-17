# SENet-MegEngine

The  MegEngine implementation of  SENet(Squeeze-and-Excitation Networks)

## Useage

### Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't want to compare the ouput error between the MegEngine implementation and PyTorch one, just ignore requirements.txt and install MegEngine from the command line:

```bash
python3 -m pip install --upgrade pip 
python3 -m pip install megengine -f https://megengine.org.cn/whl/mge.html
```

### Convert Weights

Convert trained weights from torch to megengine, the converted weights will be save in ./pretained/ .

```bash
python convert_weights.py -m  resnet154
```

If  the download  speed is too slow, you  may download them manually.

### Compare

Use `python compare.py` .

By default, the compare script will convert the torch state_dict to the format that megengine need.

If you want to compare the error by checkpoints, you neet load them manually.

### Load From Hub

Import from megengine.hub:

Way 1:

```python
import megengine.module as M
from megengine import hub

modelhub = hub.import_module(
    repo_info='asthestarsfalll/SENet-MegEngine:main', git_host='github.com')

# load SENet and custom on you own
model = modelhub.SENet(block=SEBottleneck, layers=[3, 8, 36, 3], groups=64, reduction=16,
	downsample_kernel_size=3, downsample_padding=1,  inplanes=128, input_3x3=True)

# load pretrained model
pretrained_model = modelhub.senet154(pretrained=True)
```

Way 2:

```python
from  megengine import hub

# load pretrained model 
model_name = 'senet154'
pretrained_model = hub.load(
    repo_info='asthestarsfalll/SENet-MegEngine:main', entry=model_name, git_host='github.com', pretrained=True)
```

You can load the model without pretrained weights like this:

```python
model = modelhub.mae_vit_large_patch16()
# or
model_name = 'seresnext101_32x4d'
model = hub.load(
    repo_info='asthestarsfalll/SENet-MegEngine:main', entry=model_name, git_host='github.com')
```

## Reference

[The timm implementation](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/senet.py)