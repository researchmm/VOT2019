import torch
import torch.nn as nn
import models.models.backbone as backbones
import models.models.bbreg as bbmodels
from models.admin import model_constructor
from collections import OrderedDict


class AASnet(nn.Module):
    def __init__(self, rgb_feature_extractor, t_feature_extractor, rgb_bb_regressor, t_bb_regressor, iou_guess, bb_regressor_layer, extractor_grad=True):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(AASnet, self).__init__()

        self.rgb_feature_extractor = rgb_feature_extractor
        self.t_feature_extractor = t_feature_extractor
        self.rgb_bb_regressor = rgb_bb_regressor
        self.t_bb_regressor = t_bb_regressor
        self.iou_guess = iou_guess
        self.bb_regressor_layer = bb_regressor_layer

        # pytorch 0.4
        # if not extractor_grad:
        #     for p in self.rgb_feature_extractor.parameters():
        #         p.requires_grad_(False)
        #     for p in self.t_feature_extractor.parameters():
        #         p.requires_grad_(False)
        #
        #     # fix iou head(except for iou fc layer)
        #     for p in self.rgb_bb_regressor.parameters():
        #         p.requires_grad_(False)
        #     for p in self.t_bb_regressor.parameters():
        #         p.requires_grad_(False)
        #
        #     # unfix iou fc
        #     for p in self.iou_guess.parameters():
        #         p.requires_grad_(True)

        if not extractor_grad:
            for p in self.rgb_feature_extractor.parameters():
                p.requires_grad = False
            for p in self.t_feature_extractor.parameters():
                p.requires_grad = False

            # fix iou head(except for iou fc layer)
            for p in self.rgb_bb_regressor.parameters():
                p.requires_grad = False
            for p in self.t_bb_regressor.parameters():
                p.requires_grad = False

            # unfix iou fc
            for p in self.iou_guess.parameters():
                p.requires_grad = False


    def forward(self, rgb_train_imgs, rgb_test_imgs, t_train_imgs, t_test_imgs,  rgb_train_bb, rgb_test_proposals, t_train_bb, t_test_proposals):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        train_imgs: template images
        test_imgs: instance images
        """
        num_sequences = rgb_train_imgs.shape[-4]
        num_train_images = rgb_train_imgs.shape[0] if rgb_train_imgs.dim() == 5 else 1
        num_test_images = rgb_test_imgs.shape[0] if rgb_test_imgs.dim() == 5 else 1

        # merge bboxes from proposals
        test_proposals = torch.cat((rgb_test_proposals, t_test_proposals), dim=1)  # TODO: check size
        train_bb = rgb_train_bb   # rgb_trian_bb and t_train_bb is the same
        # train_bb = torch.cat((rgb_train_bb, t_train_bb), dim=0)  # TODO: check size

        # Extract backbone features
        rgb_train_feat = self.extract_backbone_features(
            rgb_train_imgs.view(-1, rgb_train_imgs.shape[-3], rgb_train_imgs.shape[-2], rgb_train_imgs.shape[-1]), flag='RGB')
        rgb_test_feat = self.extract_backbone_features(
            rgb_test_imgs.view(-1, rgb_test_imgs.shape[-3], rgb_test_imgs.shape[-2], rgb_test_imgs.shape[-1]), flag='RGB')

        t_train_feat = self.extract_backbone_features(
            t_train_imgs.view(-1, t_train_imgs.shape[-3], t_train_imgs.shape[-2], t_train_imgs.shape[-1]), flag='T')
        t_test_feat = self.extract_backbone_features(
            t_test_imgs.view(-1, t_test_imgs.shape[-3], t_test_imgs.shape[-2], t_test_imgs.shape[-1]), flag='T')

        # For clarity, send the features to bb_regressor in sequence form, i.e. [sequence, batch, feature, row, col]
        rgb_train_feat_iou = [feat.view(num_train_images, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1])
                          for feat in rgb_train_feat.values()]
        rgb_test_feat_iou = [feat.view(num_test_images, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1])
                         for feat in rgb_test_feat.values()]

        t_train_feat_iou = [feat.view(num_train_images, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1])
                              for feat in t_train_feat.values()]
        t_test_feat_iou = [feat.view(num_test_images, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1])
                             for feat in t_test_feat.values()]

        # get pooling feature from rgb/t extract
        feature_rgb, batch_size, num_proposals_per_batch, num_test_images = self.rgb_bb_regressor(rgb_train_feat_iou, rgb_test_feat_iou, train_bb.view(num_train_images, num_sequences, 4), test_proposals.view(num_train_images, num_sequences, -1, 4))
        feature_t, _, _, _ = self.t_bb_regressor(t_train_feat_iou, t_test_feat_iou, train_bb.view(num_train_images, num_sequences, 4), test_proposals.view(num_train_images, num_sequences, -1, 4))

        # Obtain iou prediction
        iou_pred = self.iou_guess(feature_rgb, feature_t, batch_size, num_proposals_per_batch, num_test_images)
        return iou_pred

    def extract_backbone_features(self, im, layers=None, flag='RGB'):
        # RGB or T
        if layers is None:
            layers = self.bb_regressor_layer

        if flag == 'RGB': return self.rgb_feature_extractor(im, layers)
        elif flag == 'T': return self.t_feature_extractor(im, layers)
        else: raise ValueError('no this kind feature extractor, check your codes')

    def extract_features(self, im, layers, flag):
        if flag == 'RGB':
            return self.rgb_feature_extractor(im, layers)
        elif flag == 'T':
            return self.t_feature_extractor(im, layers)
        else:
            raise ValueError('no this kind feature extractor, check your codes')




# @model_constructor
def aas_resnet50(iou_input_dim=(256,256), iou_inter_dim=(256,256), rgb_pretrained_path=None, t_pretrained_path=None, backbone_pretrained=None):
    # backbone
    # # for train
    # rgb_backbone_net = backbones.resnet50(pretrained_path=rgb_pretrained_path)
    # t_backbone_net = backbones.resnet50(pretrained_path=t_pretrained_path)

    # only for test
    rgb_backbone_net = backbones.resnet50(pretrained_path=None)
    t_backbone_net = backbones.resnet50(pretrained_path=None)

    # extract features
    rgb_extract = bbmodels.RGBIoUNet(input_dim=(128*4,256*4), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)
    t_extract = bbmodels.TIoUNet(input_dim=(128*4,256*4), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # iou pretict
    iou_predictor = bbmodels.iou_predictor(pred_inter_dim=iou_inter_dim)

    # load iou feature extracter
    # if rgb_pretrained_path is not None and t_pretrained_path is not None:
    if False:  # only for test
        rgb_extract, t_extract = load_iou_extractor(rgb_extract, t_extract, rgb_pretrained_path, t_pretrained_path)


    net = AASnet(rgb_feature_extractor=rgb_backbone_net, t_feature_extractor=t_backbone_net, rgb_bb_regressor=rgb_extract, t_bb_regressor=t_extract, iou_guess= iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                  extractor_grad=False)

    return net


def load_iou_extractor(rgb_extract, t_extract, rgb_pretrained_path, t_pretrained_path):
    """

    :param rgb_extract:
    :param t_extract:
    :param rgb_pretrained_path:
    :param t_pretrained_path:
    :return: load layers in bbregressor (except for fc layer)
    """
    device = torch.cuda.current_device()

    print("load RGB IOU net feature extractor")
    pretrained_dict = torch.load(rgb_pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    new_dict = OrderedDict()
    for key in pretrained_dict['net'].keys():
        if key.startswith('bb_regressor'):
            new_dict[key.split('bb_regressor.')[-1]] = pretrained_dict['net'][key]
        elif key.startswith('feature_extractor'):
            pass
        else:
            raise ValueError("unknown key")
            # new_dict[key] = pretrained_dict['net'][key]

    rgb_extract = load_pretrain(rgb_extract, new_dict)

    print("load T IOU net feature extractor")
    pretrained_dict = torch.load(t_pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    new_dict = OrderedDict()
    for key in pretrained_dict['net'].keys():
        if key.startswith('bb_regressor'):
            new_dict[key.split('bb_regressor.')[-1]] = pretrained_dict['net'][key]
        elif key.startswith('feature_extractor'):
            pass
        else:
            raise ValueError("unknown key")

    t_extract = load_pretrain(t_extract, new_dict)

    return rgb_extract, t_extract


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('missing keys:{}'.format(sorted(missing_keys)))

    print('=========================================')
    print('unused checkpoint keys:{}'.format(sorted(unused_pretrained_keys)))
    # print('used keys:{}'.format(used_pretrained_keys))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def load_pretrain(model, pretrained_dict):

    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
