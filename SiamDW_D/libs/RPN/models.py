import torch.nn as nn

from libs.RPN.resnet_atrous import resnet50
from libs.RPN.neck import AdjustAllLayer
from libs.RPN.head import MultiRPN

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = resnet50(used_layers=[2, 3, 4])

        # build neck
        # self.neck = AdjustAllLayer([512, 1024, 2048], [128, 256, 512])
        self.neck = AdjustAllLayer([512, 1024, 2048], [256, 256, 256])

        # channels = [128, 256, 512]
        channels = [256, 256, 256]

        self.rpn_head = MultiRPN(5, channels, True)

    def template(self, z):
        zf = self.backbone(z)
        zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        return cls, loc
