from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from collections import OrderedDict
from libs.models.backbone.resnext101_64x4d_features import resnext101_64x4d_features

__all__ = ['ResNeXt101_64x4d', 'resnext101_64x4d']

pretrained_settings = {
    'resnext101_64x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/resnext101_64x4d-e77a0586.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    }
}

class ResNeXt101_64x4d(nn.Module):

    def __init__(self, output_layers=None):
        super(ResNeXt101_64x4d, self).__init__()
        self.num_classes = 1000
        self.features = resnext101_64x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, 1000)
        self.output_layers = output_layers

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None):
        outputs = OrderedDict()
        output_layers = self.output_layers

        for i in range(8):
            x = self.features[i](x)
            if i>=4 and self._add_output_and_check('layer{}'.format(i - 3), x, outputs, output_layers):
                return outputs

        raise ValueError('Error in output layers when forward')

def resnext101_64x4d(output_layers=None, pretrained=False, path=None):
    if output_layers is None:
        raise ValueError('Error in output layers')
    else:
        for l in output_layers:
            if l not in ['conv0', 'layer1', 'layer2', 'layer3', 'layer4']:
                raise ValueError('Unknown layer: {}'.format(l))
    model = ResNeXt101_64x4d(output_layers)
    if pretrained:
        import torch
        if path == None:
            model.load_state_dict(model_zoo.load_url(pretrained_settings['resnext101_64x4d']['imagenet']['url']))
        else:
            model.load_state_dict(torch.load(path))
    return model
