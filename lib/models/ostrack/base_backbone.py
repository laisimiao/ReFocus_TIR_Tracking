from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.ostrack.utils import combine_tokens, recover_tokens


class BaseBackbone(nn.Module):

    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.return_stage = cfg.MODEL.RETURN_STAGES
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG

        # resize patch embedding
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # for patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        # for cls token (keep it but not used)
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        # separate token and segment token
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        # self.cls_token = None
        # self.pos_embed = None

        if self.return_inter:
            for i_layer in self.return_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def first_feedforward(self, z, x, return_attention=False, ffd_output_layer=12, reuse_layer=0):
        if reuse_layer is not None and reuse_layer > ffd_output_layer:
            reuse_layer = ffd_output_layer

        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        mid_output = None
        for i, blk in enumerate(self.blocks):
            if return_attention:
                x, attn = blk(x, return_attention=True)
            else:
                x = blk(x, return_attention=False)

            if reuse_layer is not None and i == (reuse_layer - 1):
                mid_output = x
            if ffd_output_layer is not None and i == (ffd_output_layer - 1) and i < (len(self.blocks)-1):
                # default ffd_output_layer is 12, will not end with break
                break

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None,
                    "mid_output": mid_output}

        return self.norm(x), aux_dict

    def second_feedforward(self, z, x, tds=None, return_attention=False, reuse_layer=0, first_mid_out=None):
        if tds is None:
            tds = [None] * len(self.blocks)
        assert isinstance(tds, list), "tds must be a list!"

        if reuse_layer > 0:
            assert first_mid_out is not None, "first_mid_out must be not None when reuse_layer > 0"
            x = first_mid_out
        else:
            B, H, W = x.shape[0], x.shape[2], x.shape[3]

            x = self.patch_embed(x)
            z = self.patch_embed(z)

            if self.add_cls_token:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                cls_tokens = cls_tokens + self.cls_pos_embed

            z += self.pos_embed_z
            x += self.pos_embed_x

            if self.add_sep_seg:
                x += self.search_segment_pos_embed
                z += self.template_segment_pos_embed

            x = combine_tokens(z, x, mode=self.cat_mode)
            if self.add_cls_token:
                x = torch.cat([cls_tokens, x], dim=1)

            x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if reuse_layer > 0 and i < reuse_layer:
                continue
            if return_attention:
                x, attn = blk(x, tds[i], return_attention=True)
            else:
                x = blk(x, tds[i], return_attention=False)


        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None}

        return self.norm(x), aux_dict

        # out, aux_dict = self.forward_features(z, x, td=top_down_signal, return_attention=return_attention)
        # return out, aux_dict

    def vanilla_feature_selection(self, first_forward_output):
        out = first_forward_output
        cos_sim = F.normalize(out, dim=-1) @ F.normalize(self.refocus_td_task_embed[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        out = out * mask
        out = out @ self.refocus_td_transform
        return out


    def feature_selection(self, first_forward_output, no_channel=False, no_spatial=False):
        out = first_forward_output
        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        out_z = out[:, :lens_z]
        agr_z = torch.mean(out_z, dim=1, keepdim=True)  # B, 1, C
        max_z = torch.max(out_z, dim=1, keepdim=True)[0]  # B, 1, C
        ## token selection
        if not no_spatial:
            spatial_specific_token = self.refocus_td_task_embed * agr_z
            cos_sim = F.normalize(out, dim=-1) @ F.normalize(spatial_specific_token.transpose(-1, -2), dim=1)  # B, N, 1
            mask = cos_sim.clamp(0, 1)
            out = out * mask
        ## channel selection
        if not no_channel:
            channel_specific_token = max_z @ self.refocus_td_transform
            out = out * channel_specific_token

        return out

    def get_td(self, feature_selection_out):
        """update for ablation: late"""
        tds = []
        for depth in range(len(self.refocus_decoders) - 1, -1, -1):
            opt = self.refocus_decoders[depth](feature_selection_out)  # out,
            tds = [opt] + tds
        if len(tds) < len(self.blocks):
            diff = len(self.blocks) - len(tds)
            tds = [None] * diff + tds
        return tds

    def forward_features_refocus(self, z, x, return_attention=False, ffd_output_layer=12, reuse_td=False,
                                 reuse_td_interval=1, test_frame_idx=1):
        # first feedforward
        reuse_layer = len(self.blocks) - len(self.refocus_decoders)  # if can enter into `forward_features_refocus`, self must has refocus_decoders
        first_ffd_output, aux_dict = self.first_feedforward(z, x, return_attention=return_attention, ffd_output_layer=ffd_output_layer, reuse_layer=reuse_layer)
        mid_output = aux_dict["mid_output"]

        # feature selection
        out = self.feature_selection(first_ffd_output)
        # out = self.vanilla_feature_selection(first_ffd_output)

        # Feedback
        if reuse_td:
            assert isinstance(reuse_td_interval, int) and reuse_td_interval > 0
            if test_frame_idx == 1:
                # the first frame need to predict
                self.tds = self.get_td(out)
            elif test_frame_idx % reuse_td_interval == 1:
                # update top-down signal when reaching interval
                self.tds = self.get_td(out)
        else:
            self.tds = self.get_td(out)

        # if test_frame_idx == 1:
        #     print(f"self.tds: {self.tds}")
        # elif test_frame_idx == 5:
        #     print(f"self.tds: {self.tds}")

        # second feedforward
        x, aux_dict = self.second_feedforward(z, x, self.tds, return_attention=return_attention, reuse_layer=reuse_layer, first_mid_out=mid_output)  # same input
        return x, aux_dict

    def forward(self, z, x, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        return_attention = kwargs.get('return_attention', False)
        ffd_output_layer = kwargs.get('ffd_output_layer', 12)

        reuse_td = kwargs.get('reuse_td', False)
        test_frame_idx = kwargs.get('test_frame_idx', 1)
        reuse_td_interval = kwargs.get('reuse_td_interval', 1)
        if 'ReFocus' in self.prompt_type:
            x, aux_dict = self.forward_features_refocus(z, x, return_attention=return_attention, ffd_output_layer=ffd_output_layer,
                                                        reuse_td=reuse_td, test_frame_idx=test_frame_idx, reuse_td_interval=reuse_td_interval)
        else:
            x, aux_dict = self.first_feedforward(z, x, return_attention=return_attention)

        return x, aux_dict



