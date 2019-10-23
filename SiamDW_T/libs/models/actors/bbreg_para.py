from . import BaseActor
import torch


class AASParaActor(BaseActor):
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        iou_pred = self.net(data['rgb_train_images'].squeeze(), data['rgb_test_images'].squeeze(), data['t_train_images'].squeeze(),
                            data['t_test_images'].squeeze(), data['rgb_train_anno'].squeeze(),
                            data['rgb_test_proposals'].squeeze(), data['t_train_anno'].squeeze(), data['t_test_proposals'].squeeze())

        iou_pred = iou_pred.view(-1, iou_pred.shape[2])

        temp_gt = torch.cat((data['rgb_proposal_iou'], data['t_proposal_iou']), dim=-1)
        iou_gt = temp_gt.view(-1, temp_gt.shape[2])
        # iou_gt = data['proposal_iou'].view(-1, data['proposal_iou'].shape[2])

        # Compute loss
        loss = self.objective(iou_pred, iou_gt)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss.item()}

        return loss, stats