import math

from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond

from lib.test.tracker.vis_utils import vis_attn_maps, vis_feat_maps


class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.yaml_name = params.yaml_name
        self.use_visdom = 0
        self.frame_id = 0
        self.vis_attn = 0
        self.return_attention = self.vis_attn
        self.vis_feat = 0

        if self.debug:
            if not self.use_visdom:
                prj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..')
                # self.save_dir = os.path.join(prj_path, "debug")
                self.save_dir = '/media/lz/T7Shield/attention_vis'
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict, seqname=None):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        self.seq_name = seqname if seqname is not None else 'vis_seq'
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        if self.debug:
            if self.vis_attn:
                self.add_attn_hook()
            if self.vis_feat:
                self.add_feat_hook()

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z,
                return_attention=self.return_attention, ffd_output_layer=self.cfg.PROMPT.REFOCUS_FFD_OUTPUT_LAYER,
                reuse_td = self.cfg.TEST.REUSE_TD, reuse_td_interval = self.cfg.TEST.REUSE_TD_INTERVAL,
                test_frame_idx=self.frame_id
            )  # add param for early !!

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug and self.frame_id % 1 == 0:
            if not self.use_visdom:

                if self.vis_attn:
                    vis_attn_maps(self.enc_attn_weights, 8, os.path.join(self.save_dir, self.seq_name), self.frame_id, last_only=False)
                if self.vis_feat:
                    vis_feat_maps(self.backbone_feature[-1], self.box_head_score[-1], 8, os.path.join(self.save_dir, self.seq_name), self.frame_id, self.yaml_name)
                # x1, y1, w, h = self.state
                # image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                # cv2.imshow('debug', image_BGR)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                # cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break
        if self.debug:
            if self.vis_attn:
                self.remove_attn_hook()
            if self.vis_feat:
                self.remove_feat_hook()


        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_attn_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        self.attn_hooks = []
        for i in range(12):
            self.attn_hooks.append(self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            ))

        self.enc_attn_weights = enc_attn_weights

    def add_feat_hook(self):
        backbone_feature, box_head_score = [], []
        self.backbone_feature_hooks = []
        self.box_head_score_hooks = []
        self.backbone_feature_hooks.append(self.network.backbone.register_forward_hook(
            lambda self, input, output: backbone_feature.append(output[0])
        ))
        self.box_head_score_hooks.append(self.network.box_head.register_forward_hook(
            lambda self, input, output: box_head_score.append(output[0])
        ))
        self.backbone_feature = backbone_feature
        self.box_head_score = box_head_score

    def remove_attn_hook(self):
        for hook in self.attn_hooks:
            hook.remove()
        self.enc_attn_weights = []

    def remove_feat_hook(self):
        for hook in self.backbone_feature_hooks:
            hook.remove()
        for hook in self.box_head_score_hooks:
            hook.remove()
        self.backbone_feature = []
        self.box_head_score = []


def get_tracker_class():
    return OSTrack
