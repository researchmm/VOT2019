import torch.nn as nn


def constant_init(module, val, bias=0):
    nn.init.constant(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform(module.weight, gain=gain)
    else:
        nn.init.xavier_normal(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform(
            module.weight, a=a, mode=mode)
    else:
        nn.init.kaiming_normal(
            module.weight, a=a, mode=mode)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant(module.bias, bias)


def caffe2_xavier_init(module, bias=0):
    # `XavierFill` in Caffe2 corresponds to `kaiming_uniform_` in PyTorch
    # Acknowledgment to FAIR's internal code
    kaiming_init(
        module,
        a=1,
        mode='fan_in',
        nonlinearity='leaky_relu',
        distribution='uniform')
