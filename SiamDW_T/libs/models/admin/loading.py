import torch
import os
import sys
from pathlib import Path
import importlib
from models.models.bbreg.aas import aas_resnet50


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('missing keys:{}'.format(missing_keys))

    print('=========================================')
    # clean it to no batch_tracked key words
    unused_pretrained_keys = [k for k in unused_pretrained_keys if 'num_batches_tracked' not in k]

    print('unused checkpoint keys:{}'.format(unused_pretrained_keys))
    # print('used keys:{}'.format(used_pretrained_keys))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True



def load_pretrain(model, pretrained_dict):

    device = torch.cuda.current_device()

    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def load_network(network_dir=None, checkpoint=None, constructor_fun_name=None, constructor_module=None, **kwargs):
        """Loads a network checkpoint file.

        Can be called in two different ways:
            load_checkpoint(network_dir):
                Loads the checkpoint file given by the path. I checkpoint_dir is a directory,
                it tries to find the latest checkpoint in that directory.
            load_checkpoint(network_dir, checkpoint=epoch_num):
                Loads the network at the given epoch number (int).

        The extra keyword arguments are supplied to the network constructor to replace saved ones.
        """


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
        net = aas_resnet50()
        checkpoint_dict = torch_load_legacy(checkpoint_path)

        net = load_pretrain(net, checkpoint_dict['net'])


        # net.constructor = checkpoint_dict['constructor']
        # if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
        #     net.info = checkpoint_dict['net_info']

        return net, checkpoint_dict


def load_weights(net, path, strict=True):
    checkpoint_dict = torch.load(path)
    weight_dict = checkpoint_dict['net']
    net.load_state_dict(weight_dict, strict=strict)
    return net


def torch_load_legacy(path):
    """Load network with legacy environment."""
    # libs.core3 load
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

    # debug
    # save_ckpt = {}
    # for k in checkpoint_dict.keys():
    #     if k == 'net':
    #         save_ckpt[k] = checkpoint_dict[k]
    #
    # torch.save(save_ckpt, '/data/home/v-zhipeng/project/codes2/vot_rgbt_release/rgbt_tracking/sharenet.pth.tar')

    # Cleanup legacy
    _cleanup_legacy_env()

    return checkpoint_dict


def _setup_legacy_env():
    importlib.import_module('models')
    sys.modules['dlframework'] = sys.modules['models']
    sys.modules['dlframework.common'] = sys.modules['models']
    for m in ('model_constructor', 'stats', 'settings', 'local'):
        importlib.import_module('models.admin.'+m)
        sys.modules['dlframework.common.utils.'+m] = sys.modules['models.admin.'+m]


def _cleanup_legacy_env():
    del_modules = []
    for m in sys.modules.keys():
        if m.startswith('dlframework'):
            del_modules.append(m)
    for m in del_modules:
        del sys.modules[m]
