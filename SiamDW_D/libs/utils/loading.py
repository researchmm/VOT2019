import os
import sys
import torch
import importlib
import libs.models.modules.net as models

from pathlib import Path


def load_network(network_dir=None, checkpoint=None, constructor_fun_name=None, constructor_module=None, net_name=None, **kwargs):
        if network_dir is not None:
            net_path = Path(network_dir)
        else:
            net_path = None

        if net_path.is_file():
            checkpoint = str(net_path)

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(net_path.glob('*.pth.tar'))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                raise Exception('No matching checkpoint file found')
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_list = sorted(net_path.glob('*_ep{:04d}.pth.tar'.format(checkpoint)))
            if not checkpoint_list or len(checkpoint_list) == 0:
                raise Exception('No matching checkpoint file found')
            if len(checkpoint_list) > 1:
                raise Exception('Multiple matching checkpoint files found')
            else:
                checkpoint_path = checkpoint_list[0]
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        checkpoint_dict = torch_load_legacy(checkpoint_path)

        # import pdb; pdb.set_trace()
        net_name = checkpoint_dict['net_name']
        net = getattr(models, net_name)
        net = net()

        net.load_state_dict(checkpoint_dict['net'], strict=False)
        print("loading net: {}".format(net_name))

        return net, checkpoint_dict


def load_weights(net, path, strict=True):
    checkpoint_dict = torch.load(path)
    weight_dict = checkpoint_dict['net']
    net.load_state_dict(weight_dict, strict=strict)
    return net


def torch_load_legacy(path):
    try:
        torch._utils._rebuild_tensor_v2
    except AttributeError:
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor
        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

    # Setup legacy env (for older networks)
    _setup_legacy_env()

    # Load network
    checkpoint_dict = torch.load(path)

    # Cleanup legacy
    _cleanup_legacy_env()

    return checkpoint_dict


def _setup_legacy_env():
    importlib.import_module('libs')
    sys.modules['dlframework'] = sys.modules['libs']
    sys.modules['dlframework.common'] = sys.modules['libs']

    importlib.import_module('libs.utils.settings')
    sys.modules['dlframework.common.utils.settings'] = sys.modules['libs.utils.settings']

def _cleanup_legacy_env():
    del_modules = []
    for m in sys.modules.keys():
        if m.startswith('dlframework'):
            del_modules.append(m)
    for m in del_modules:
        del sys.modules[m]

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    #print('missing keys:{}'.format(missing_keys))
    #print('unused checkpoint keys:{}'.format(unused_pretrained_keys))
    #print('used keys:{}'.format(used_pretrained_keys))
    #print('missing keys:{}'.format(len(missing_keys)))
    #print('unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    #print('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    '''
    Old style model is stored with all names of parameters share common prefix 'module.'
    '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model_name, pretrained_path):
    model = getattr(models, model_name)
    model = model()

    # print('load pretrained model from {}'.format(pretrained_path))
    try:
        torch._utils._rebuild_tensor_v2
    except AttributeError:
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor
        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
