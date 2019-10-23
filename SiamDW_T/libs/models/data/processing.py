import torch
import torchvision.transforms as transforms
from libs.core import TensorDict
import libs.models.data.processing_utils as prutils


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), train_transform=None, test_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if train_transform or
                                test_transform is None.
            train_transform - The set of transformations to be applied on the train images. If None, the 'transform'
                                argument is used instead.
            test_transform  - The set of transformations to be applied on the test images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the train and test images.  For
                                example, it can be used to convert both test and train images to grayscale.
        """
        self.transform = {'train': transform if train_transform is None else train_transform,
                          'test':  transform if test_transform is None else test_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class AASProcessing(BaseProcessing):
    """ The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. A set of proposals are then generated for the test images by jittering the ground truth box.

    """
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, proposal_params,
                 mode='pair', *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.proposal_params = proposal_params
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """
        if 'train' in mode: mode = 'train'
        if 'test' in mode: mode = 'test'

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * self.center_jitter_factor[mode]).item()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box

        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        proposals = torch.zeros((num_proposals, 4))
        gt_iou = torch.zeros(num_proposals)

        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(box, min_iou=self.proposal_params['min_iou'],
                                                             sigma_factor=self.proposal_params['sigma_factor']
                                                             )

        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -

        returns:
            TensorDict - output data block with following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
                'test_proposals'-
                'proposal_iou'  -
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            num_train_images = len(data['rgb_train_images'])
            # zzp: to keep rgb and t transform align
            # all_images = data['rgb_train_images'] + data['rgb_test_images'] + data['t_train_images'] + data['t_test_images']
            # all_images_trans = self.transform['joint'](*all_images)
            #
            # data['rgb_train_images'] = all_images_trans[:num_train_images]
            # data['t_train_images'] = all_images_trans[num_train_images*2:num_train_images*3]
            # data['rgb_test_images'] = all_images_trans[num_train_images: num_train_images*2]
            # data['t_test_images'] = all_images_trans[num_train_images*3:]

            rgb_all_images = data['rgb_train_images'] + data['rgb_test_images']
            t_all_images = data['t_train_images'] + data['t_test_images']
            rgb_all_images_trans = self.transform['joint'](*rgb_all_images)
            t_all_images_trans = self.transform['joint'](*t_all_images)

            data['rgb_train_images'] = rgb_all_images_trans[:num_train_images]
            data['t_train_images'] = t_all_images_trans[:num_train_images]
            data['rgb_test_images'] = rgb_all_images_trans[num_train_images:]
            data['t_test_images'] = t_all_images_trans[num_train_images:]

        for s in ['rgb_train', 'rgb_test', 't_train', 't_test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            # zzp make rgb and t have same jitter
            if 'rgb_train' in s:
                jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
                temp_train_anno = jittered_anno
            if 'rgb_test' in s:
                jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
                temp_test_anno = jittered_anno
            if 't_train' in s:
                jittered_anno = temp_train_anno
            if 't_test' in s:
                jittered_anno = temp_test_anno

            # Crop image region centered at jittered_anno box
            # print('{}: jittanno: {}, bbox: {}'.format(s + '_images', jittered_anno, data[s+'_anno']))
            crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                self.search_area_factor, self.output_sz)

            # Apply transforms
            data[s + '_images'] = [self.transform[s.split('_')[-1]](x) for x in crops]
            data[s + '_anno'] = boxes

        # Generate proposals
        rgb_frame2_proposals, rgb_gt_iou = zip(*[self._generate_proposals(a) for a in data['rgb_test_anno']])
        t_frame2_proposals, t_gt_iou = zip(*[self._generate_proposals(a) for a in data['t_test_anno']])

        data['rgb_test_proposals'] = list(rgb_frame2_proposals)
        data['rgb_proposal_iou'] = list(rgb_gt_iou)

        data['t_test_proposals'] = list(t_frame2_proposals)
        data['t_proposal_iou'] = list(t_gt_iou)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(prutils.stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data