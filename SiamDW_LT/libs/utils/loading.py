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
