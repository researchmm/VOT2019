import torch.nn as nn
import libs.models.modules as bbmodels
import libs.models.backbone as backbones


class STAR_A(nn.Module):
    def __init__(self, feature_extractor, bb_regressor, bb_regressor_layer, extractor_grad=True):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(STAR_A, self).__init__()

        self.feature_extractor = feature_extractor
        self.bb_regressor = bb_regressor
        self.bb_regressor_layer = bb_regressor_layer

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

    def forward(self, train_imgs, test_imgs, train_bb, test_proposals):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        num_sequences = train_imgs.shape[-4]
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1
        num_test_images = test_imgs.shape[0] if test_imgs.dim() == 5 else 1

        # Extract backbone features
        train_feat = self.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = self.extract_backbone_features(
            test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # For clarity, send the features to bb_regressor in sequence form, i.e. [sequence, batch, feature, row, col]
        train_feat_iou = [feat.view(num_train_images, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1])
                          for feat in train_feat.values()]
        test_feat_iou = [feat.view(num_test_images, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1])
                         for feat in test_feat.values()]

        # Obtain iou prediction
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou,
                                     train_bb.view(num_train_images, num_sequences, 4),
                                     test_proposals.view(num_train_images, num_sequences, -1, 4))
        return iou_pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)

# def resnet50_512(pred_input_dim=(256,256), pred_inter_dim=(256,256), backbone_libs=True, path=None):
def atom_resnet50_512(pred_input_dim=(256,256), pred_inter_dim=(256,256), backbone_libs=True, path=None):
    input_dim = [512, 1024]
    pred_input_dim = [512, 512]
    pred_mid_dim = [512, 512]
    pred_inter_dim = [512, 512]

    # backbone
    backbone_net = backbones.resnet50(libs=backbone_libs, path=path)

    # Bounding box regressor
    iou_predictor = bbmodels.IoUNet(input_dim=input_dim,
                                        pred_input_dim=pred_input_dim,
                                        pred_mid_dim=pred_mid_dim,
                                        pred_inter_dim=pred_inter_dim)

    net = STAR_A(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                  extractor_grad=False)

    return net

# def resnext101_64_dropout(iou_input_dim=(256,256), iou_inter_dim=(256,256), backbone_libs=True, path=None):
def atom_resnext101_64_dropout(iou_input_dim=(256,256), iou_inter_dim=(256,256), backbone_pretrained=True, path=None):
    output_layers = ['layer2', 'layer3']
    channel = [512, 1024]

    # backbone
    backbone_net = backbones.resnext101_64x4d(output_layers=output_layers, pretrained=backbone_pretrained, path=path)

    # Bounding box regressor
    iou_predictor = bbmodels.IoUNet_dropout(input_dim=(channel[0], channel[1]), pred_input_dim=iou_input_dim,
                                        pred_inter_dim=iou_inter_dim)

    net = STAR_A(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=output_layers,
                  extractor_grad=False)

    return net
