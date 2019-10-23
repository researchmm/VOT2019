import torch
import torch.nn.functional as F
import torch.nn
import math
import time
import numpy as np
import cv2
import os
import os.path as osp
from torch.autograd import Variable
from libs.utils import dcf, fourier, TensorList, operation
from libs.features.preprocessing import numpy_to_torch
from libs.utils.plotting import show_tensor
from libs.utils.optimization import GaussNewtonCG, ConjugateGradient, GradientDescentL2
from .optim import ConvProblem, FactorizedConvProblem
from libs.features import augmentation
from libs.core.base import BaseTracker

# add fpn lib
import libs.mmcv as mmcv
from libs.mmcv.runner import load_checkpoint
from libs.FPNlib.mmdet.models import build_detector
from libs.FPNlib.mmdet.apis import inference_detector, show_result

# add others
from libs.RPN.rpnpp_utils import *
from libs.RPN.models import ModelBuilder
from libs.RPN.model_load import load_pretrain
from .processing_utils import centered_crop
from .fpn_helper import proposal_filter, window_simi, cos_similarity

class ResnextFar(BaseTracker):

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.features.initialize(model_name='resnext101_64')
        self.features_initialized = True

    def python2round(self, f):
        # return round(f)
        if round(f + 1) - round(f) != 1:
            return f + abs(f) / f * 0.5
        return round(f)

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        return im_patch

    def SiamRPN_init(self, img, bbox, model):
        self.model = model

        self.center_pos = np.array([bbox[0], bbox[1]])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + 0.5 * np.sum(self.size)
        h_z = self.size[1] + 0.5 * np.sum(self.size)
        # s_z = round(np.sqrt(w_z * h_z))
        s_z = self.python2round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos, 127, s_z, self.channel_average)
        self.model.template(Variable(z_crop.unsqueeze(0)).cuda())

    def generate_anchor(self, score_size):
        anchors = Anchors(8, [0.33, 0.5, 1, 2, 3], [8])
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def SiamRPN_track(self, img):

        w_z = self.size[0] + 0.5 * np.sum(self.size)
        h_z = self.size[1] + 0.5 * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = 127. / s_z

        if self.longterm_state: instance_size = 831
        else: instance_size = 255

        score_size = (instance_size - 127) // 8 + 1 + 8
        hanning = np.hanning(score_size)
        window = np.outer(hanning, hanning)
        window = np.tile(window.flatten(), 5)
        anchors = self.generate_anchor(score_size)

        s_x = s_z * (instance_size / 127.)
        x_crop = self.get_subwindow(img, self.center_pos, instance_size,
                                    self.python2round(s_x), self.channel_average)

        score, delta = self.model.track(Variable(x_crop.unsqueeze(0)).cuda())
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchors[:, 2] + anchors[:, 0]
        delta[1, :] = delta[1, :] * anchors[:, 3] + anchors[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchors[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchors[:, 3]

        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(delta[2, :], delta[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
        # ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (delta[2, :] / delta[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.PENALTY_K)
        pscore = penalty * score

        # window
        if not self.longterm_state and not self.flag_map:
            pscore = pscore * (1 - self.WINDOW_INFLUENCE) + window * self.WINDOW_INFLUENCE
        else:
            pscore = pscore * (1 - 0.001) + window * 0.001
        best_idx = np.argmax(pscore)

        bbox = delta[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * self.LR

        best_score = score[best_idx]
        # if best_score >= self.CONFIDENCE_LOW:
        if True:
            cx = bbox[0] + self.center_pos[0]
            cy = bbox[1] + self.center_pos[1]

            width = self.size[0] * (1 - lr) + bbox[2] * lr
            height = self.size[1] * (1 - lr) + bbox[3] * lr
        else:
            cx = self.center_pos[0]
            cy = self.center_pos[1]

            width = self.size[0]
            height = self.size[1]

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        cx, cy, width, height = bbox_clip(cx, cy, width, height, img.shape[:2])

        rpn_track_bbox = [cx - width / 2,
                         cy - height / 2,
                         width,
                         height]

        return best_score, rpn_track_bbox

    def SiamRPN_cropsz(self, target_sz):
        wc_z = target_sz[1] + 0.5 * sum(target_sz)
        hc_z = target_sz[0] + 0.5 * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = 127 / s_z
        if self.longterm_state: i_sz = 831
        else: i_sz = 255
        d_search = (i_sz - 127) / 2.
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        return int(s_x)

    def avg_depth(self, depth_im, bbox):
        # get center around depth
        cx, cy = bbox[0]+bbox[2]/2, bbox[1] + bbox[2]/2

        def shrink_box(bbox, mh, mw, r=0.4):
            w, h = (1-r) * bbox[2], (1-r) * bbox[3]
            if w<2 or h<2:
                lx = max(cx-w/2, 0)
                ly = max(cy-h/2, 0)
                return [lx, ly, bbox[2], bbox[3]]
            else:
                lx = max(cx-w/2, 0)
                ly = max(cy-h/2, 0)
                w = min(w, mw - w)
                h = min(h, mh - h)
                return [lx, ly, w, h]

        try:
            h, w = depth_im.shape
        except:
            print ("depth_im is Error!")
            print (depth_im)
            return -1.
        # print ("h: {} w: {}".format(h, w))
        shrink_box = shrink_box(bbox, h, w, r=0.4)
        shrink_box = np.asarray(shrink_box).astype(np.int16)
        avg_depth = np.mean(depth_im[shrink_box[1]:(shrink_box[1]+shrink_box[3]), \
                shrink_box[0]:(shrink_box[0]+shrink_box[2])])

        if np.isnan(avg_depth):
            return -1.
        return avg_depth


    def initialize_rgbd(self, raw_im, image, depth_im, state, init_online=False):
        # Initialize SiamRPN
        self.longterm_state = False
        self.flag_map = False
        model = ModelBuilder()
        main_path = self.params.main_path
        path = osp.join(main_path, "networks/RPN/siamrpn.pth")
        model = load_pretrain(model, path).cuda().eval()
        self.SiamRPN_init(raw_im.copy(), state, model)
        # RPN Config
        # self.PENALTY_K = 0.24
        self.PENALTY_K = 0.119
        # self.WINDOW_INFLUENCE = 0.5
        self.WINDOW_INFLUENCE = 0.48
        # self.LR = 0.25
        self.LR = 0.349
        self.CONFIDENCE_LOW = 0.7
        # self.CONFIDENCE_HIGH = 0.9995
        self.CONFIDENCE_HIGH = 0.9999

        # some flag
        self.only_rpn = False
        self.split_model = True

        # Initialize some stuff
        self.init_online = init_online
        self.read_pretrained = True
        self.frame_num = 1
        if not hasattr(self.params, 'device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # initialize FPN
        self.fpn_cfg = mmcv.Config.fromfile(osp.join(main_path, '../libs/FPNlib/configs/rpn_r50_fpn_1x.py'))
        checkpoint = osp.join(main_path, 'networks/FPN/rpn_r50_fpn_2x.pth')
        self.fpn_cfg.model.pretrained = None
        # BGR input
        self.fpn_model = build_detector(self.fpn_cfg.model, test_cfg=self.fpn_cfg.test_cfg)
        load_checkpoint(self.fpn_model, checkpoint, map_location='cpu')
        # torch.cuda.set_device(self.params.device)
        self.fpn_model.cuda()
        self.fpn_model.eval()

        # add history dict save dict a {'state', 'score', 'f_iounet'}
        # depth image info
        self.record = {'avg_similarity':[], 'predict_iou': [], 'filter_score': [], 'raw_score': [], 'distractor_score': []}
        self.history_gap = 10
        self.history_save_thres = 0.5
        # bbox: save all bboxes
        # fc_features: save only good features as template
        init_bbox = state.copy()
        init_bbox[0] = init_bbox[0] - 0.5 * init_bbox[2]
        init_bbox[1] = init_bbox[1] - 0.5 * init_bbox[3]
        d_init = self.avg_depth(depth_im, init_bbox)
        self.history_info = {'counter': 0, 'bbox_counter':[0], 'bboxes':[init_bbox], 'distractor_bbox':[], 'fc_feature_counter': [], 'fc_features': [], 'depth': [d_init]}

        # add atom score margin
        self.lost_thres = 0.15
        self.trust_thres = 0.5

        self.area_factor = 1.1
        self.sample_size = 288

        # Initialize features
        self.initialize_features()

        # Check if image is color
        self.params.features.set_is_color(image.shape[2] == 3)

        # Get feature specific params
        self.fparams = self.params.features.get_fparams('feature_params')

        self.time = 0
        tic = time.time()

        # Get position and size
        self.pos = torch.Tensor([state[1], state[0]])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Set search area
        self.target_scale = 1.0
        search_area = torch.prod(self.target_sz * self.params.search_area_scale)
        if search_area > self.params.max_image_sample_size:
            self.target_scale = math.sqrt(search_area / self.params.max_image_sample_size)
        elif search_area < self.params.min_image_sample_size:
            self.target_scale = math.sqrt(search_area / self.params.min_image_sample_size)

        # Check if IoUNet is used
        self.use_iou_net = getattr(self.params, 'use_iou_net', True)

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Use odd square search area and set sizes
        feat_max_stride = max(self.params.features.stride())
        if getattr(self.params, 'search_area_shape', 'square') == 'square':
            area = torch.prod(self.base_target_sz * self.params.search_area_scale)
            self.img_sample_sz = torch.round(torch.sqrt(torch.Tensor([area]))) * torch.ones(2)
        elif self.params.search_area_shape == 'initrect':
            self.img_sample_sz = torch.round(self.base_target_sz * self.params.search_area_scale)
        else:
            raise ValueError('Unknown search area shape')
        if self.params.feature_size_odd:
            self.img_sample_sz += feat_max_stride - self.img_sample_sz % (2 * feat_max_stride)
        else:
            self.img_sample_sz += feat_max_stride - (self.img_sample_sz + feat_max_stride) % (2 * feat_max_stride)

        # Set sizes
        self.img_support_sz = self.img_sample_sz
        self.feature_sz = self.params.features.size(self.img_sample_sz)
        self.output_sz = self.params.score_upsample_factor * self.img_support_sz  # Interpolated size of the output
        self.kernel_size = self.fparams.attribute('kernel_size')

        self.iou_img_sample_sz = self.img_sample_sz

        # Optimization options
        self.params.precond_learning_rate = self.fparams.attribute('learning_rate')
        if self.params.CG_forgetting_rate is None or max(self.params.precond_learning_rate) >= 1:
            self.params.direction_forget_factor = 0
        else:
            self.params.direction_forget_factor = (1 - max(
                self.params.precond_learning_rate)) ** self.params.CG_forgetting_rate

        self.output_window = None
        if getattr(self.params, 'window_output', False):
            if getattr(self.params, 'use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(),
                                                        self.output_sz.long() * self.params.effective_search_area / self.params.search_area_scale,
                                                        centered=False).cuda()
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=False).cuda()

        # Initialize some learning things
        self.init_learning()

        # Convert image
        im = numpy_to_torch(image)
        self.im = im  # For debugging only

        # Setup scale bounds
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        x = self.generate_init_samples(im)

        # Initialize iounet
        if self.use_iou_net:
            self.init_iou_net()

        # Initialize projection matrix
        self.init_projection_matrix(x)

        # Transform to get the training sample
        train_x = self.preprocess_sample(x)

        # Generate label function
        init_y = self.init_label_function(train_x)

        # Init memory
        self.init_memory(train_x)

        # Init optimizer and do initial optimization
        self.init_optimization(train_x, init_y)

        self.pos_iounet = self.pos.clone()

        self.time += time.time() - tic

    def init_optimization(self, train_x, init_y):
        # Initialize filter
        filter_init_method = getattr(self.params, 'filter_init_method', 'zeros')
        if self.read_pretrained:
            self.load_online_filter()
        elif self.init_online:
            self.filter = TensorList(
                [torch.zeros(1, cdim, sz[0], sz[1]).cuda() for x, cdim, sz in zip(train_x, self.compressed_dim, self.kernel_size)])
        if filter_init_method == 'zeros':
            pass
        elif filter_init_method == 'randn':
            for f in self.filter:
                f.normal_(0, 1/f.numel())
        else:
            raise ValueError('Unknown "filter_init_method"')

        # Get parameters
        self.params.update_projection_matrix = getattr(self.params, 'update_projection_matrix', True) and self.params.use_projection_matrix
        optimizer = getattr(self.params, 'optimizer', 'GaussNewtonCG')

        # Setup factorized joint optimization
        if self.params.update_projection_matrix:
            self.joint_problem = FactorizedConvProblem(self.init_training_samples, init_y, self.filter_reg,
                                                       self.fparams.attribute('projection_reg'), self.params, self.init_sample_weights,
                                                       self.projection_activation, self.response_activation)

            # Variable containing both filter and projection matrix
            joint_var = self.filter.concat(self.projection_matrix)

            # Initialize optimizer
            analyze_convergence = getattr(self.params, 'analyze_convergence', False)
            if optimizer == 'GaussNewtonCG':
                self.joint_optimizer = GaussNewtonCG(self.joint_problem, joint_var.variable(), plotting=(self.params.debug >= 3), analyze=analyze_convergence, fig_num=(12, 13, 14))
            elif optimizer == 'GradientDescentL2':
                self.joint_optimizer = GradientDescentL2(self.joint_problem, joint_var, self.params.optimizer_step_length, self.params.optimizer_momentum, plotting=(self.params.debug >= 3), debug=analyze_convergence, fig_num=(12, 13))

            # Do joint optimization
            if isinstance(self.params.init_CG_iter, (list, tuple)):
                self.joint_optimizer.run(self.params.init_CG_iter)
            else:
                self.joint_optimizer.run(self.params.init_CG_iter // self.params.init_GN_iter, self.params.init_GN_iter)


        # Re-project samples with the new projection matrix
        compressed_samples = self.project_sample(self.init_training_samples,
                                                 self.projection_matrix.variable()).tensor()
        for train_samp, init_samp in zip(self.training_samples, compressed_samples):
            train_samp[:init_samp.shape[0], ...] = init_samp

        self.hinge_mask = None

        # Initialize optimizer
        self.conv_problem = ConvProblem(self.training_samples, self.y, self.filter_reg, self.sample_weights,
                                        self.response_activation)

        if optimizer == 'GaussNewtonCG':
            self.filter_optimizer = ConjugateGradient(self.conv_problem, self.filter,
                                                      fletcher_reeves=self.params.fletcher_reeves,
                                                      direction_forget_factor=self.params.direction_forget_factor,
                                                      debug=(self.params.debug >= 3), fig_num=(12, 13))
        elif optimizer == 'GradientDescentL2':
            self.filter_optimizer = GradientDescentL2(self.conv_problem, self.filter, self.params.optimizer_step_length,
                                                      self.params.optimizer_momentum, debug=(self.params.debug >= 3),
                                                      fig_num=12)

        # Transfer losses from previous optimization
        if self.params.update_projection_matrix:
            self.filter_optimizer.residuals = self.joint_optimizer.residuals
            self.filter_optimizer.losses = self.joint_optimizer.losses

        if not self.params.update_projection_matrix:
            self.filter_optimizer.run(self.params.init_CG_iter)

        # Post optimization
        self.filter_optimizer.run(self.params.post_init_CG_iter)

        # Free memory
        del self.init_training_samples
        if self.params.use_projection_matrix:
            del self.joint_problem, self.joint_optimizer

    def get_state(self, new_pos):
        inside_ratio = 0.2
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        return torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)

    ########## added by hongyuan ##########
    def get_bbox(self, new_pos, new_scale = None):
        # Update scale
        if new_scale is not None:
            target_scale = np.clip(new_scale, self.min_scale_factor, self.max_scale_factor)
            target_sz = self.base_target_sz * target_scale

        # Update pos
        inside_ratio = 0.2
        inside_offset = (inside_ratio - 0.5) * target_sz
        pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)
        # limit target not to go out pic so much
        new_state = torch.cat((pos[[1,0]] - (target_sz[[1,0]]-1)/2, target_sz[[1,0]]))
        return new_state.tolist()

    def load_online_filter(self):
        # print ("load filter!")
        self.filter = torch.load(self.params.read_filter_path)
        self.filter = TensorList(self.filter)

    def load_online_proj(self):
        # print ("load proj!")
        self.projection_matrix = torch.load(self.params.read_proj_path)
        self.projection_matrix = TensorList(self.projection_matrix)

    def map_search(self):
        init_target_pos = self.center_pos.tolist()
        init_target_sz = self.size.tolist()

        crop_sz = self.SiamRPN_cropsz(self.size)
        w, h = self.img.shape[1], self.img.shape[0]
        stax, stay = init_target_pos[0], init_target_pos[1]
        X, Y = [], []
        crop_sz = getattr(self.params, 'crop_map_ratio', 0.7) * crop_sz
        # append x
        for i in range(1000):
            if stax - i * crop_sz < 0 and stax + i * crop_sz > w: break
            if stax - i * crop_sz >= 0: X.append(stax - i * crop_sz)
            if stax + i * crop_sz <= w and i > 0: X.append(stax + i * crop_sz)

        # append y
        for i in range(1000):
            if stay - i * crop_sz < 0 and stay + i * crop_sz > h: break
            if stay - i * crop_sz >= 0: Y.append(stay - i * crop_sz)
            if stay + i * crop_sz <= h and i > 0: Y.append(stay + i * crop_sz)

        RPN_status, RPN_box = [], []
        ATOM_status, ATOM_box = [], []
        self.flag_map = True

        for x in X:
            for y in Y:
                # RPN map search
                target_pos = np.array([x, y])
                target_sz = np.array(init_target_sz)
                self.center_pos = target_pos
                self.size = target_sz
                score, bbox = self.SiamRPN_track(self.img.copy())
                RPN_status.append(score)
                RPN_box.append(bbox)

                # ATOM map search
                sample_pos = torch.Tensor([x, y])
                sample_scales = self.target_scale * self.params.scale_factors
                test_x = self.extract_processed_sample(self.im.clone(), sample_pos, sample_scales, self.img_sample_sz)
                scores_raw = self.apply_filter(test_x)
                _, translation_vec, _, _, _, scale_ind, s , flag = \
                    self.localize_target(scores_raw)
                mx_score = torch.max(s[scale_ind, ...])
                ATOM_status.append(mx_score)
                atom_pos = self.get_state(sample_pos + translation_vec).tolist()
                atom_sz = self.target_sz.tolist()
                ATOM_box.append([atom_pos[1] - atom_sz[1] / 2.,
                                 atom_pos[0] - atom_sz[0] / 2.,
                                 atom_sz[1], atom_sz[0]])

        # RPN back to init status
        self.center_pos = np.array(init_target_pos)
        self.size = np.array(init_target_sz)
        self.flag_map = False

        # ATOM back to init status
        sample_pos = self.pos.round()
        sample_scales = self.target_scale * self.params.scale_factors
        _ = self.extract_processed_sample(self.im.clone(), sample_pos, sample_scales, self.img_sample_sz)

        return RPN_status, RPN_box, ATOM_status, ATOM_box

    def check_ATOM(self, status):
        pool = []
        for idx, item in enumerate(status):
            if item > getattr(self.params, 'possible_thr', 0.05):
                pool.append(idx)
        return pool

    def check_RPN(self, status):
        pool = []
        for idx, item in enumerate(status):
            if item >= self.CONFIDENCE_LOW:
                pool.append(idx)
        return pool

    def get_union(self, abox, rbox):
        return [min(abox[0], rbox[0]),
                min(abox[1], rbox[1]),
                max(abox[2], rbox[2]),
                max(abox[3], rbox[3])]

    def valid_depth(self, depth_a, history_depth):
        depth_margin = 600
        if depth_a>2000:
            depth_percent = depth_margin / depth_a
        else:
            depth_percent = 0.4
        d = np.abs((depth_a-history_depth))
        if d == 0 or depth_a == 0:
            return False
        p = d/depth_a
        # if d>depth_margin or p > depth_percent:
        if p > depth_percent:
            return False
        else:
            return True

    def track(self, raw_im, image, depth_im):
        saved_im = image
        self.frame_num += 1
        self.img = raw_im.copy() # bgr

        # Convert image
        im = numpy_to_torch(image)
        self.im = im    # For debugging only

        # SiamRPN first analyse
        rpn_score, rpn_bbox = self.SiamRPN_track(raw_im.copy())
        self.longterm_state = False
        if self.only_rpn:
            return rpn_bbox, rpn_score

        # Get sample
        sample_pos = self.pos.round()
        sample_scales = self.target_scale * self.params.scale_factors
        test_x = self.extract_processed_sample(Variable(im), self.pos, sample_scales, self.img_sample_sz)

        scores_raw = self.apply_filter(test_x)
        mx_score, translation_vec, distractor_score, distractor_translation_vec, filter_score, scale_ind, s, flag = self.localize_target(scores_raw)
        atom_score = mx_score
        np_distractor_score = float(distractor_score)

        init_bbox = self.get_bbox(sample_pos + translation_vec, sample_scales[int(scale_ind)])
        distractor_bbox = self.get_bbox(sample_pos + distractor_translation_vec, sample_scales[int(scale_ind)])

        # Update position and scale
        if flag != 'not_found' or self.frame_num==2:
            if self.use_iou_net:
                update_scale_flag = getattr(self.params, 'update_scale_when_uncertain', True) or flag != 'uncertain'
                if getattr(self.params, 'use_classifier', True):
                    self.update_state(sample_pos + translation_vec)

                predicted_iou, temp_fc_feature = self.refine_target_box(sample_pos, sample_scales[int(scale_ind)], scale_ind, update_scale_flag)

            elif getattr(self.params, 'use_classifier', True):
                self.update_state(sample_pos + translation_vec, sample_scales[int(scale_ind)])

            history_bboxes = self.history_info['bboxes'] # may be usefull, e.g. Motion state estimation
            history_fc_features = self.history_info['fc_features']
            if len(history_fc_features)>5:
                history_fc_features = history_fc_features[-5:]
            if len(history_bboxes) == 0:
                history_bbox = []
            else:
                history_bbox = history_bboxes[-1]
            np_predicted_iou = predicted_iou
            try:
                np_fc_feature = temp_fc_feature.cpu().numpy().squeeze()
                avg_simi = window_simi(init_bbox, np_fc_feature, history_bbox, history_fc_features)
            except:
                avg_simi = 0.5
        else:
            np_predicted_iou = -2.
            avg_simi = -2.
        self.record['avg_similarity'].append(avg_simi)
        self.record['filter_score'].append(filter_score)
        self.record['predict_iou'].append(np_predicted_iou)
        self.record['raw_score'].append(atom_score)
        self.record['distractor_score'].append(np_distractor_score)

        # get atom bbox
        # Set the pos of the tracker to iounet pos
        if self.use_iou_net and flag != 'not_found':
            self.pos = self.pos_iounet.clone()
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))
        atom_bbox = new_state.tolist()

        # get avg depth
        history_depth = self.history_info['depth']
        d_rpn = self.avg_depth(depth_im, rpn_bbox)
        d_atom = self.avg_depth(depth_im, atom_bbox)
        valid_d_rpn = self.valid_depth(d_rpn, history_depth[-1])
        valid_d_atom = self.valid_depth(d_atom, history_depth[-1])

        update_flag = flag not in ['not_found', 'uncertain']
        depth_flag = ~valid_d_atom and ~valid_d_rpn and (self.frame_num>300)

        if atom_score < self.lost_thres and self.history_info['counter']>0:
            if rpn_score<self.CONFIDENCE_LOW:
                target_pos = self.center_pos.tolist()
                target_sz = self.size.tolist()

                self.pos = torch.Tensor([target_pos[1], target_pos[0]])
                self.target_sz = torch.Tensor([target_sz[1], target_sz[0]])

                RPN_status, RPN_box, ATOM_status, ATOM_box = self.map_search()

                RPN_idx, ATOM_idx = self.check_RPN(RPN_status), \
                                    self.check_ATOM(ATOM_status)

                if len(RPN_idx) <= 0 and len(ATOM_idx) <= 0:
                    return [target_pos[0] - target_sz[0] / 2.,
                            target_pos[1] - target_sz[1] / 2.,
                            target_sz[0], target_sz[1]], mx_score

                mx_iou, mx_rid, mx_aid = 0, 0, 0

                for r_id in RPN_idx:
                    for a_id in ATOM_idx:
                        iou = self.IoU(RPN_box[r_id], ATOM_box[a_id])
                        if iou > mx_iou:
                            mx_iou, mx_rid, mx_aid = iou, r_id, a_id

                if mx_iou > getattr(self.params, 'iou_thr', 0.1):
                    box = self.get_union(RPN_box[mx_rid], ATOM_box[mx_aid])
                    d_ar = self.avg_depth(depth_im, box)
                    valid_d_ar = self.valid_depth(d_ar, self.history_info['depth'][-1])
                    if valid_d_ar:
                        self.history_info['depth'].append(d_ar)

                    box[0], box[1] = box[0] + box[2] / 2., box[1] + box[3] / 2.
                    # restart ATOM sz
                    self.pos = torch.Tensor([box[1], box[0]])
                    self.target_sz = torch.Tensor([box[3], box[2]])

                else:
                    # print ("Use FPN")
                    img = cv2.cvtColor(raw_im, cv2.COLOR_RGB2BGR)
                    result, f_list = inference_detector(self.fpn_model, img, self.fpn_cfg)
                    proposals = result[:, :4].copy()
                    proposals[:, 2] = proposals[:, 2] - proposals[:, 0]
                    proposals[:, 3] = proposals[:, 3] - proposals[:, 1]
                    score1, pos, target_sz = self.redetect(proposals, img)
                    if score1 > 0.25:
                        # restart ATOM sz
                        self.pos = torch.from_numpy(pos.copy())
                        self.target_sz = torch.from_numpy(target_sz.copy())
                    else:
                        # restart ATOM sz
                        box = rpn_bbox.copy()
                        box[0], box[1] = box[0] + box[2] / 2., box[1] + box[3] / 2.
                        self.pos = torch.Tensor([box[1], box[0]])
                        self.target_sz = torch.Tensor([box[3], box[2]])

            if valid_d_atom:
                self.history_info['depth'].append(d_atom)
                new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]])).tolist()
                output_bbox = new_state
                # output_bbox = atom_bbox
            else:
                new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]])).tolist()
                output_bbox = new_state
        else:
            if valid_d_atom:
                self.history_info['depth'].append(d_atom)
                output_bbox = atom_bbox

            else:
                output_bbox = atom_bbox

        if rpn_score>self.CONFIDENCE_HIGH:
            output_bbox = rpn_bbox
            box = output_bbox.copy()
            box[0], box[1] = box[0] + box[2] / 2., box[1] + box[3] / 2.
            # restart ATOM sz
            self.pos = torch.Tensor([box[1], box[0]])
            self.target_sz = torch.Tensor([box[3], box[2]])

        # Check flags and set learning rate if hard negative
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.hard_negative_learning_rate if hard_negative else None

        # if update_flag and avg_simi>self.trust_thres:
        if update_flag:
            train_x = TensorList([x[int(scale_ind):int(scale_ind) + 1, ...] for x in test_x])
            train_y = self.get_label_function(sample_pos, sample_scales[int(scale_ind)])
            self.update_memory(train_x, train_y, learning_rate)

        # Train filter
        if hard_negative:
            self.filter_optimizer.run(self.params.hard_negative_CG_iter)
            # self.save_online()
        elif (self.frame_num-1) % self.params.train_skipping == 0:
            self.filter_optimizer.run(self.params.CG_iter)
            # self.save_online()

        if self.history_info['counter'] == 0:
            self.history_info['fc_feature_counter'].append(self.history_info['counter'])
            self.history_info['fc_features'].append(temp_fc_feature.cpu().numpy().squeeze())
        elif update_flag and avg_simi > self.history_save_thres and (self.history_info['counter'] - self.history_info['fc_feature_counter'][-1]) > self.history_gap==0:
            self.history_info['fc_feature_counter'].append(self.history_info['counter'])
            self.history_info['fc_features'].append(temp_fc_feature.cpu().numpy().squeeze())

        self.history_info['bbox_counter'].append(self.history_info['counter'])
        self.history_info['bboxes'].append(output_bbox)
        self.history_info['distractor_bbox'].append(distractor_bbox)
        self.history_info['counter'] += 1

        if self.split_model:
            if atom_score<0.2 and avg_simi<0.3 and rpn_score>self.CONFIDENCE_LOW:
                return rpn_bbox, rpn_score

        return output_bbox, (atom_score+rpn_score)/2.

    def apply_filter(self, sample_x: TensorList):
        return operation.conv2d(sample_x, self.filter.variable(), mode='same')

    def localize_target(self, scores_raw):
        # Weighted sum (if multiple features) with interpolation in fourier domain
        weight = self.fparams.attribute('translation_weight', 1.0)
        scores_raw = weight * scores_raw

        test_scores = scores_raw[0].data.squeeze().cpu().numpy()
        filter_score = np.max(test_scores)

        sf_weighted = fourier.cfft2(scores_raw.tensor()) / (scores_raw.size(2) * scores_raw.size(3))
        for i, (sz, ksz) in enumerate(zip(self.feature_sz, self.kernel_size)):
            sf_weighted[i] = fourier.shift_fs(sf_weighted[i], math.pi * (1 - torch.Tensor([ksz[0]%2, ksz[1]%2]) / sz))

        scores_fs = fourier.sum_fs(sf_weighted)
        scores = fourier.sample_fs(scores_fs, self.output_sz)

        if self.output_window is not None and not getattr(self.params, 'perform_hn_without_windowing', False):
            scores *= self.output_window

        if getattr(self.params, 'advanced_localization', False):
            return self.localize_advanced(scores)

        # Get maximum
        max_score, max_disp = dcf.max2d(scores)
        # print(max_score)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp.float().cpu()

        # Convert to displacements in the base scale
        disp = (max_disp + self.output_sz / 2) % self.output_sz - self.output_sz / 2

        # Compute translation vector and scale change factor
        translation_vec = disp[scale_ind, ...].view(-1) * (self.img_support_sz / self.output_sz) * self.target_scale
        translation_vec *= self.params.scale_factors[scale_ind]

        # Shift the score output for visualization purposes
        if self.params.debug >= 2:
            sz = scores.shape[-2:]
            scores = torch.cat([scores[...,sz[0]//2:,:], scores[...,:sz[0]//2,:]], -2)
            scores = torch.cat([scores[...,:,sz[1]//2:], scores[...,:,:sz[1]//2]], -1)

        return max_score, translation_vec, max_score, translation_vec, filter_score, scale_ind, scores, None

    def localize_advanced(self, scores):
        """Does the advanced localization with hard negative detection and target not found."""

        sz = scores.shape[-2:]

        if self.output_window is not None and getattr(self.params, 'perform_hn_without_windowing', False):
            scores_orig = scores.clone()

            scores_orig = torch.cat([scores_orig[..., (sz[0] + 1) // 2:, :], scores_orig[..., :(sz[0] + 1) // 2, :]], -2)
            scores_orig = torch.cat([scores_orig[..., :, (sz[1] + 1) // 2:], scores_orig[..., :, :(sz[1] + 1) // 2]], -1)

            scores *= self.output_window

        test_scores = scores.squeeze().cpu().numpy()
        filter_score = np.max(test_scores)

        # Shift scores back
        scores = torch.cat([scores[...,(sz[0]+1)//2:,:], scores[...,:(sz[0]+1)//2,:]], -2)
        scores = torch.cat([scores[...,:,(sz[1]+1)//2:], scores[...,:,:(sz[1]+1)//2]], -1)

        # Find maximum
        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - self.output_sz / 2
        translation_vec1 = target_disp1 * (self.img_support_sz / self.output_sz) * self.target_scale

        max_score1 = float(max_score1)

        if bool(float(max_score1 < self.params.target_not_found_threshold)):
            return max_score1, translation_vec1, max_score1, translation_vec1, filter_score, scale_ind, scores, 'not_found'

        if self.output_window is not None and getattr(self.params, 'perform_hn_without_windowing', False):
            scores = scores_orig

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * self.target_sz / self.target_scale
        tneigh_top = max(round(max_disp1[0] - target_neigh_sz[0] / 2), 0)
        tneigh_bottom = min(round(max_disp1[0] + target_neigh_sz[0] / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1] - target_neigh_sz[1] / 2), 0)
        tneigh_right = min(round(max_disp1[1] + target_neigh_sz[1] / 2 + 1), sz[1])
        scores_masked = scores[int(scale_ind):int(scale_ind)+1,...].clone()
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - self.output_sz / 2
        translation_vec2 = target_disp2 * (self.img_support_sz / self.output_sz) * self.target_scale
        max_score2 = float(max_score2)
        # Handle the different cases
        if bool(float(max_score2 > self.params.distractor_threshold * max_score1)):
            disp_norm1 = math.sqrt(torch.sum(target_disp1**2))
            disp_norm2 = math.sqrt(torch.sum(target_disp2**2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return max_score1, translation_vec1, max_score2, translation_vec2, filter_score,  scale_ind, scores, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return max_score2, translation_vec2, max_score1, translation_vec1, filter_score, scale_ind, scores, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return max_score1, translation_vec1, max_score2, translation_vec2, filter_score, scale_ind, scores, 'uncertain'

            # If also the distractor is close, return with highest score
            return max_score1, translation_vec1, max_score2, translation_vec2, filter_score, scale_ind, scores, 'uncertain'

        if bool(float(max_score2 > self.params.hard_negative_threshold * max_score1)) \
                and bool(float(max_score2 > self.params.target_not_found_threshold)):
            return max_score1, translation_vec1, max_score2, translation_vec2, filter_score, scale_ind, scores, 'hard_negative'

        return max_score1, translation_vec1, max_score2, translation_vec2, filter_score, scale_ind, scores, None

    def extract_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        return self.params.features.extract(im, pos, scales, sz)

    def get_iou_features(self):
        return self.params.features.get_unique_attribute('iounet_features')

    def get_iou_backbone_features(self):
        return self.params.features.get_unique_attribute('iounet_backbone_features')

    def extract_processed_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor) -> (TensorList, TensorList):
        x = self.extract_sample(im, pos, scales, sz)
        # print ("x: {}".format(x.shape))
        return self.preprocess_sample(self.project_sample(x))

    def preprocess_sample(self, x: TensorList) -> (TensorList, TensorList):
        if getattr(self.params, '_feature_window', False):
            x = x * self.feature_window
        return x

    def project_sample(self, x: TensorList, proj_matrix = None):
        # Apply projection matrix
        if proj_matrix is None:
            proj_matrix = self.projection_matrix.variable()
        return operation.conv2d(x, proj_matrix).apply(self.projection_activation)

    def init_learning(self):
        # Get window function
        self.feature_window = TensorList([dcf.hann2d(sz).cuda() for sz in self.feature_sz])

        # Filter regularization
        self.filter_reg = self.fparams.attribute('filter_reg')

        # Activation function after the projection matrix (phi_1 in the paper)
        projection_activation = getattr(self.params, 'projection_activation', 'none')
        if isinstance(projection_activation, tuple):
            projection_activation, act_param = projection_activation

        if projection_activation == 'none':
            self.projection_activation = lambda x: x
        elif projection_activation == 'relu':
            self.projection_activation = torch.nn.ReLU(inplace=True)
        elif projection_activation == 'elu':
            self.projection_activation = torch.nn.ELU(inplace=True)
        elif projection_activation == 'mlu':
            self.projection_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')

        # Activation function after the output scores (phi_2 in the paper)
        response_activation = getattr(self.params, 'response_activation', 'none')
        if isinstance(response_activation, tuple):
            response_activation, act_param = response_activation

        if response_activation == 'none':
            self.response_activation = lambda x: x
        elif response_activation == 'relu':
            self.response_activation = torch.nn.ReLU(inplace=True)
        elif response_activation == 'elu':
            self.response_activation = torch.nn.ELU(inplace=True)
        elif response_activation == 'mlu':
            self.response_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')


    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Generate augmented initial samples."""

        # Compute augmentation size
        aug_expansion_factor = getattr(self.params, 'augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift operator
        get_rand_shift = lambda: None
        random_shift_factor = getattr(self.params, 'random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor).long().tolist()

        # Create transofmations
        self.transforms = [augmentation.Identity(aug_output_sz)]
        if 'shift' in self.params.augmentation:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz) for shift in self.params.augmentation['shift']])
        if 'relativeshift' in self.params.augmentation:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz) for shift in self.params.augmentation['relativeshift']])
        if 'fliplr' in self.params.augmentation and self.params.augmentation['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in self.params.augmentation:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in self.params.augmentation['blur']])
        if 'scale' in self.params.augmentation:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in self.params.augmentation['scale']])
        if 'rotate' in self.params.augmentation:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in self.params.augmentation['rotate']])

        # Generate initial samples
        init_samples = self.params.features.extract_transformed(im, self.pos, self.target_scale, aug_expansion_sz, self.transforms)

        # Remove augmented samples for those that shall not have
        for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
            if not use_aug:
                init_samples[i] = init_samples[i][0:1, ...]

        # Add dropout samples
        if 'dropout' in self.params.augmentation:
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1]*num)
            for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
                if use_aug:
                    init_samples[i] = torch.cat([init_samples[i], F.dropout2d(init_samples[i][0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        return init_samples

    def init_projection_matrix(self, x):
        # Set if using projection matrix
        self.params.use_projection_matrix = getattr(self.params, 'use_projection_matrix', True)

        if self.params.use_projection_matrix:
            self.compressed_dim = self.fparams.attribute('compressed_dim', None)

            proj_init_method = getattr(self.params, 'proj_init_method', 'pca')
            if proj_init_method == 'pca':
                x_mat = TensorList([e.permute(1, 0, 2, 3).reshape(e.shape[1], -1).clone() for e in x])
                x_mat -= x_mat.mean(dim=1, keepdim=True)
                cov_x = x_mat @ x_mat.t()
                if self.read_pretrained:
                    self.load_online_proj()
                elif self.init_online:
                    self.projection_matrix = TensorList(
                        [None if cdim is None else torch.svd(C)[0][:, :cdim].t().unsqueeze(-1).unsqueeze(-1).clone() for C, cdim in
                        zip(cov_x, self.compressed_dim)])
            elif proj_init_method == 'randn':
                if self.read_pretrained:
                    self.load_online_proj()
                elif self.init_online:
                    self.projection_matrix = TensorList(
                        [None if cdim is None else torch.zeros(cdim, ex.shape[1], 1, 1).normal_(0, 1 / math.sqrt(
                            ex.shape[1])).cuda() for ex, cdim in
                        zip(x, self.compressed_dim)])
        else:
            self.compressed_dim = x.size(1)
            if self.read_pretrained:
                self.load_online_proj()
            elif self.init_online:
                self.projection_matrix = TensorList([None]*len(x))

    def init_label_function(self, train_x):
        # Allocate label function
        self.y = TensorList(
            [torch.zeros(self.params.sample_memory_size, 1, x.shape[2], x.shape[3]).cuda() for x in train_x])

        # Output sigma factor
        output_sigma_factor = self.fparams.attribute('output_sigma_factor')
        self.sigma = math.sqrt(torch.prod(
            (self.feature_sz / self.img_support_sz * self.base_target_sz)[0])) * output_sigma_factor * torch.ones(2)

        # Center pos in normalized coords
        target_center_norm = (self.pos - self.pos.round()) / (self.target_scale * self.img_support_sz)

        # Generate label functions
        for y, sig, sz, ksz, x in zip(self.y, self.sigma, self.feature_sz, self.kernel_size, train_x):
            center_pos = sz * target_center_norm + 0.5 * torch.Tensor([(ksz[0] + 1) % 2, (ksz[1] + 1) % 2])
            for i, T in enumerate(self.transforms[:x.shape[0]]):
                sample_center = center_pos + torch.Tensor(T.shift) / self.img_support_sz * sz
                y[i, 0, ...] = dcf.label_function_spatial(sz, sig, sample_center)

        # Return only the ones to use for initial training
        return TensorList([y[:x.shape[0], ...] for y, x in zip(self.y, train_x)])

    def init_memory(self, train_x):
        # Initialize first-frame training samples
        self.num_init_samples = train_x.size(0)
        self.init_sample_weights = TensorList([torch.ones(1).cuda() / x.shape[0] for x in train_x])
        self.init_training_samples = train_x

        # Sample counters and weights
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([torch.zeros(self.params.sample_memory_size).cuda() for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, self.init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw[0]

        # Initialize memory
        self.training_samples = TensorList(
            [torch.zeros(self.params.sample_memory_size, cdim, x.shape[2], x.shape[3]).cuda() for x, cdim in
             zip(train_x, self.compressed_dim)])

    def update_memory(self, sample_x: TensorList, sample_y: TensorList, learning_rate = None):
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind,
                                                 self.num_stored_samples, self.num_init_samples, self.fparams,
                                                 learning_rate)
        self.previous_replace_ind = replace_ind
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind + 1, ...] = x.data
        for y_memory, y, ind in zip(self.y, sample_y, replace_ind):
            y_memory[ind:ind + 1, ...] = y
        if self.hinge_mask is not None:
            for m, y, ind in zip(self.hinge_mask, sample_y, replace_ind):
                m[ind:ind + 1, ...] = (y >= self.params.hinge_threshold).float()
        self.num_stored_samples += 1

    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, fparams, learning_rate = None):
        # Update weights and get index to replace in memory
        replace_ind = []
        for sw, prev_ind, num_samp, num_init, fpar in zip(sample_weights, previous_replace_ind, num_stored_samples,
                                                          num_init_samples, fparams):
            lr = learning_rate
            if lr is None:
                lr = fpar.learning_rate

            init_samp_weight = getattr(fpar, 'init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                _, r_ind = torch.min(sw[s_ind:], 0)
                r_ind = int(r_ind) + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def get_label_function(self, sample_pos, sample_scale):
        # Generate label function
        train_y = TensorList()
        target_center_norm = (self.pos - sample_pos) / (sample_scale * self.img_support_sz)
        for sig, sz, ksz in zip(self.sigma, self.feature_sz, self.kernel_size):
            center = sz * target_center_norm + 0.5 * torch.Tensor([(ksz[0] + 1) % 2, (ksz[1] + 1) % 2])
            train_y.append(dcf.label_function_spatial(sz, sig, center))
        return train_y

    def update_state(self, new_pos, new_scale = None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = 0.2
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)
        # limit target not to go out pic so much

    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates"""
        box_center = (pos - sample_pos) / sample_scale + (self.iou_img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([self.flip(target_ul, 0), self.flip(box_sz, 0)])

    def flip(self, tensor, idx):
        temp = np.flip(tensor.numpy(), idx).copy()
        tensor = torch.from_numpy(temp)
        return tensor

    def init_iou_net(self):
        # Setup IoU net
        self.iou_predictor = self.params.features.get_unique_attribute('iou_predictor')
        for p in self.iou_predictor.parameters():
            p.requires_grad = False

        # Get target boxes for the different augmentations
        self.iou_target_box = self.get_iounet_box(self.pos, self.target_sz, self.pos.round(), self.target_scale)
        target_boxes = TensorList()
        if self.params.iounet_augmentation:
            for T in self.transforms:
                if not isinstance(T, (
                augmentation.Identity, augmentation.Translation, augmentation.FlipHorizontal, augmentation.FlipVertical,
                augmentation.Blur)):
                    break
                target_boxes.append(self.iou_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        else:
            target_boxes.append(self.iou_target_box.clone())
        target_boxes = torch.cat(target_boxes.view(1, 4), 0).cuda()

        # Get iou features
        iou_backbone_features = self.get_iou_backbone_features()

        # Remove other augmentations such as rotation
        iou_backbone_features = TensorList([x[:target_boxes.shape[0], ...] for x in iou_backbone_features])

        # Extract target feat
        target_feat = self.iou_predictor.get_filter(iou_backbone_features, target_boxes)
        self.target_feat = TensorList([x.detach().mean(0) for x in target_feat])

        if getattr(self.params, 'iounet_not_use_reference', False):
            self.target_feat = TensorList([torch.full_like(tf, tf.norm() / tf.numel()) for tf in self.target_feat])

    def refine_target_box(self, sample_pos, sample_scale, scale_ind, update_scale = True):
        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features()
        iou_features = TensorList([x[int(scale_ind):int(scale_ind) + 1, ...] for x in iou_features])

        init_boxes = init_box.view(1, 4).clone()
        if self.params.num_init_random_boxes > 0:
            # Get random initial boxes
            square_box_sz = math.sqrt(init_box[2:].prod())
            rand_factor = square_box_sz * torch.cat(
                [self.params.box_jitter_pos * torch.ones(2), self.params.box_jitter_sz * torch.ones(2)])
            minimal_edge_size = init_box[2:].min() / 3
            rand_bb = (torch.rand(self.params.num_init_random_boxes, 4) - 0.5) * rand_factor
            new_sz = (init_box[2:] + rand_bb[:, 2:]).clamp(min=minimal_edge_size)
            new_center = (init_box[:2] + init_box[2:] / 2) + rand_bb[:, :2]
            init_boxes = torch.cat([new_center - new_sz / 2, new_sz], 1)
            init_boxes = torch.cat([init_box.view(1, 4), init_boxes])

        # Refine boxes by maximizing iou
        output_boxes, output_iou, temp_fc_features = self.optimize_boxes(iou_features, init_boxes)

        # *****record iou bboxes

        try:
            # Remove weird boxes with extreme aspect ratios
            output_boxes[:, 2:].clamp_(min=1)
            aspect_ratio = output_boxes[:, 2] / output_boxes[:, 3]
            keep_ind = (aspect_ratio < self.params.maximal_aspect_ratio) * (
                        aspect_ratio > 1 / self.params.maximal_aspect_ratio)
            ind = []
            for idx, item in enumerate(keep_ind):
                if int(item) == 1:
                    ind.append(idx)
            keep_ind = ind
            output_boxes = output_boxes[keep_ind, :]
            output_iou = output_iou[keep_ind]
        except:
            return 0., torch.ones(512).float().cuda()

        # If no box found
        if output_boxes.shape[0] == 0:
            return 0., torch.ones(512).float().cuda()

        # Take average of top k boxes
        k = getattr(self.params, 'iounet_k', 5)
        topk = min(k, output_boxes.shape[0])
        _, inds = torch.topk(output_iou, topk)
        predicted_box = output_boxes[inds, :].mean(0)
        predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0)
        self.predicted_iou = predicted_iou

        # Update position
        new_pos = predicted_box[:2] + predicted_box[2:] / 2 - (self.iou_img_sample_sz - 1) / 2
        new_pos = self.flip(new_pos, 0) * sample_scale + sample_pos
        new_target_sz = self.flip(predicted_box[2:], 0) * sample_scale
        new_scale = math.sqrt(new_target_sz.prod() / self.base_target_sz.prod())

        # record iou correction

        self.pos_iounet = new_pos.clone()

        if getattr(self.params, 'use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale

        # merge top3 fc_features
        inds = inds.tolist()
        temp_fc_features = temp_fc_features[inds]
        temp_fc_features = torch.mean(temp_fc_features, dim=0)
        return float(predicted_iou), temp_fc_features


    def IoU(self, rect1, rect2):
        # overlap
        import numpy as np
        x1, y1, x2, y2 = rect1[0], rect1[1], rect1[0] + rect1[2], rect1[1] + rect1[3]
        tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[0] + rect2[2], rect2[1] + rect2[3]
        xx1 = np.maximum(tx1, x1)
        yy1 = np.maximum(ty1, y1)
        xx2 = np.minimum(tx2, x2)
        yy2 = np.minimum(ty2, y2)
        ww = np.maximum(0, xx2 - xx1)
        hh = np.maximum(0, yy2 - yy1)
        area = (x2 - x1) * (y2 - y1)
        target_a = (tx2 - tx1) * (ty2 - ty1)
        inter = ww * hh
        overlap = inter / (area + target_a - inter)
        return overlap

    def optimize_boxes(self, iou_features, init_boxes):
        output_boxes = Variable(init_boxes.view(1, -1, 4).cuda())
        step_length = self.params.box_refinement_step_length

        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init = output_boxes.clone().detach()
            bb_init.requires_grad = True

            if i_ == 0:
                outputs, temp_fc_features = self.iou_predictor.predict_iou(self.target_feat, iou_features, bb_init)
            else:
                outputs, _ = self.iou_predictor.predict_iou(self.target_feat, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient=torch.ones_like(outputs))

            # Update proposal
            output_boxes = bb_init + step_length * bb_init.grad * bb_init[:, :, 2:].repeat(1, 1, 2)
            output_boxes.detach_()

            step_length *= self.params.box_refinement_step_decay

        return output_boxes.view(-1, 4).cpu().data, outputs.detach().view(-1).cpu().data, temp_fc_features.data

    ########## added by hongyuan ##########
    def find_pos(self, score_map):
        delta = 1
        h, w = score_map.shape
        score1 = np.max(score_map)

        max_loc1 = np.where(score_map == score1)
        score_loc1 = np.array([max_loc1[0][0], max_loc1[1][0]])
        xy_disp1 =  score_loc1 * 16

        mask_score_map = score_map.copy()
        lx = max(0, score_loc1[0] - delta)
        rx = min(w, score_loc1[0] + delta)
        ly = max(0, score_loc1[1] - delta)
        ry = min(h, score_loc1[1] + delta)
        mask_score_map[lx:rx, ly:ry] = 0.
        score2 = np.max(mask_score_map)

        max_loc2 = np.where(mask_score_map == score2)
        score_loc2 = np.array([max_loc2[0][0], max_loc2[1][0]])
        xy_disp2 =  score_loc2 * 16
        return score1, xy_disp1, score2, xy_disp2

    def redetect(self, proposals, img):
        # get fc_features and predict iou of all proposals
        num_boxes = len(proposals)
        images = [img for i in range(num_boxes)]
        images = np.stack(images, axis=0)

        raw_crop_imgs, crop_boxes = centered_crop(images, proposals, self.area_factor, self.sample_size)

        test_imgs = np.asarray(raw_crop_imgs)
        test_imgs = torch.from_numpy(test_imgs.transpose(0, 3, 1, 2)).float()

        _ = self.params.features.extract_patches(test_imgs)
        proposal_iounet_features = self.get_iou_features()
        target_feat = self.target_feat.unsqueeze(0).repeat(num_boxes, 1, 1, 1)
        crop_boxes = torch.stack(crop_boxes, dim=0)
        crop_boxes = Variable(crop_boxes.view(num_boxes, 1, -1).cuda())
        iounet_scores, proposal_fc_features = self.iou_predictor.predict_iou(target_feat, proposal_iounet_features, crop_boxes)

        iounet_scores = iounet_scores.data.squeeze().cpu().numpy()
        proposal_fc_features = proposal_fc_features.data.cpu().numpy()

        history_boxes = self.history_info['bboxes']
        history_fc_features = self.history_info['fc_features']
        history_len = len(history_boxes)
        if history_len > 0 and history_len < 8:
            history_boxes = np.asarray(history_boxes)
            history_fc_features = np.asarray(history_fc_features)
        elif history_len >=8:
            history_boxes = np.asarray(history_boxes[-8:])
            history_fc_features = np.asarray(history_fc_features[-8:])
        else:
            assert  NotImplementedError()

        envelope_box, rest_indexes = proposal_filter(proposals, iounet_scores, proposal_fc_features, history_fc_features, history_boxes)
        rest_proposals = proposals[rest_indexes]
        rest_fc_features = proposal_fc_features[rest_indexes]
        f_simi = []
        for i in range(len(rest_indexes)):
            f_simi.append(cos_similarity(rest_fc_features[i], history_fc_features[0]))
        rest_predict_ious = iounet_scores[rest_indexes]
        rest_rects = rest_proposals.copy()

        amp_factor = np.sqrt(np.prod(envelope_box[2:4]) / np.mean(rest_proposals[:, 2]*rest_proposals[:, 3]))
        amp_factor = np.min([np.max([amp_factor, 2]), 2])

        # crop imgae
        area_factor = self.area_factor / amp_factor
        sample_size = self.sample_size * 2
        search_img, seach_box = centered_crop([img], envelope_box.reshape(1, 4), area_factor, sample_size)
        search_img = np.asarray(search_img)
        search_img = torch.from_numpy(search_img.transpose(0, 3, 1, 2)).float()
        test_x = self.params.features.extract_patches(search_img)
        test_x = self.preprocess_sample(self.project_sample(test_x))

        # Compute scores, remove output window
        scores_raw = self.apply_filter(test_x)
        score = scores_raw[0][0, 0, :, :]
        score = score.data.cpu().numpy()

        score1, xy_disp1, score2, xy_disp2 = self.find_pos(score)
        xy_disp =  xy_disp1

        # find the proposal which is close to max_score
        rest_poses = rest_proposals.copy()[:, :2] + 0.5 * rest_proposals.copy()[:, 2:]
        envelope_pos = envelope_box[:2] + 0.5*envelope_box[2:]
        # map_poses = (rest_poses - envelope_pos) + 0.5*amp_factor*self.sample_size
        map_poses = (rest_poses - envelope_pos) * sample_size / (np.sqrt(np.prod(envelope_box[2:])) * area_factor) + 0.5 * sample_size
        dists = np.mean((map_poses - xy_disp) ** 2, axis=1)
        min_index = np.argmin(dists)

        if score1 < 0.25 and (score1 - score2) < 0.03:
            # choose nearest bbox
            history_bbox = history_boxes[-1].copy()
            history_pos = history_bbox[:2] + 0.5*history_bbox[2:]
            dists = np.mean((map_poses - xy_disp) ** 2, axis=1)
            min_index = np.argmin(dists)

            r = 0.8
            simis = r * np.asarray(f_simi) + (1-r) * np.asarray(rest_predict_ious)

            # finally use simis
            if simis[min_index] < 0.5:
                simi_idx = np.argmax(simis)
                min_index = simi_idx

        best_proposal_index = rest_indexes[min_index]
        best_proposal_iounet_feature = TensorList([f[best_proposal_index].unsqueeze(0) for f in proposal_iounet_features])
        best_crop_boxes = crop_boxes[best_proposal_index]
        best_proposal = proposals[best_proposal_index]

        best_proposal_num = len(best_proposal)
        pos = best_proposal[:2] + 0.5 * best_proposal[2:]
        pos = pos[::-1]
        target_sz = best_proposal[2:][::-1]
        return score1, pos, target_sz

