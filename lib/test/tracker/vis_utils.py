import numpy as np


############## used for visulize eliminated tokens #################
def get_keep_indices(decisions):
    keep_indices = []
    for i in range(3):
        if i == 0:
            keep_indices.append(decisions[i])
        else:
            keep_indices.append(keep_indices[-1][decisions[i]])
    return keep_indices


def gen_masked_tokens(tokens, indices, alpha=0.2):
    # indices = [i for i in range(196) if i not in indices]
    indices = indices[0].astype(int)
    tokens = tokens.copy()
    tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255
    return tokens


def recover_image(tokens, H, W, Hp, Wp, patch_size):
    # image: (C, 196, 16, 16)
    image = tokens.reshape(Hp, Wp, patch_size, patch_size, 3).swapaxes(1, 2).reshape(H, W, 3)
    return image


def pad_img(img):
    height, width, channels = img.shape
    im_bg = np.ones((height, width + 8, channels)) * 255
    im_bg[0:height, 0:width, :] = img
    return im_bg


def gen_visualization(image, mask_indices, patch_size=16):
    # image [224, 224, 3]
    # mask_indices, list of masked token indices

    # mask mask_indices need to cat
    # mask_indices = mask_indices[::-1]
    num_stages = len(mask_indices)
    for i in range(1, num_stages):
        mask_indices[i] = np.concatenate([mask_indices[i-1], mask_indices[i]], axis=1)

    # keep_indices = get_keep_indices(decisions)
    image = np.asarray(image)
    H, W, C = image.shape
    Hp, Wp = H // patch_size, W // patch_size
    image_tokens = image.reshape(Hp, patch_size, Wp, patch_size, 3).swapaxes(1, 2).reshape(Hp * Wp, patch_size, patch_size, 3)

    stages = [
        recover_image(gen_masked_tokens(image_tokens, mask_indices[i]), H, W, Hp, Wp, patch_size)
        for i in range(num_stages)
    ]
    imgs = [image] + stages
    imgs = [pad_img(img) for img in imgs]
    viz = np.concatenate(imgs, axis=1)
    return viz

###################### vis Fig.4 like in paper laizi 2023/7/6 ######################
import torch
import os
import matplotlib.pyplot as plt
import matplotlib
import math

def vis_attn_maps(attns, template_feat_size, save_path, frame_id, last_only=False):
    """
    attn: [bs, nh, lens_t+lens_s, lens_t+lens_s] * encoder_layers (e.g. 12)
    if feed forward twice, encoder_layers will be 24
    """
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#04686b", "#fcaf7c"])  # plasma
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    attn = attns[0]
    bs, hn, _, _ = attn.shape
    lens_t = template_feat_size * template_feat_size
    lens_s = attn.shape[-1] - lens_t
    search_feat_size = int(math.sqrt(lens_s))
    assert search_feat_size ** 2 == lens_s, "search_feat_size ** 2 must be equal to lens_s"
    # Default: CE_TEMPLATE_RANGE = 'CTR_POINT'
    if template_feat_size == 8:
        index = slice(3, 5)
    elif template_feat_size == 12:
        index = slice(5, 7)
    elif template_feat_size == 7:
        index = slice(3, 4)
    else:
        raise NotImplementedError


    for block_num, attn in enumerate(attns):
        if last_only:
            if block_num not in [11, 23]:
                continue
        if os.path.exists(os.path.join(save_path, f'frame{frame_id}_block{block_num + 1}_attn_weight.png')):
            print(f"-1")
            return
            # if block_num < len(attns)-1:
            #     continue
        attn_t = attn[:, :, :lens_t, lens_t:]
        box_mask_z = torch.zeros([bs, template_feat_size, template_feat_size], device=attn.device)
        box_mask_z[:, index, index] = 1
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # bs, len_s
        attn_t_plot = attn_t.squeeze(dim=0).reshape((search_feat_size, search_feat_size)).cpu().numpy()
        fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=300)
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
        ax = fig.add_subplot(111)
        ax.imshow(attn_t_plot, cmap='plasma', interpolation='nearest')
        ax.axis('off')
        plt.savefig(os.path.join(save_path, f'frame{frame_id}_block{block_num+1}_attn_weight.png'))
        plt.close()

def vis_feat_maps(backbone_out, head_score, template_feat_size, save_path, frame_id, yaml_name):
    """
    backbone_out: (B,Hz*Wz+Hx*Wx,C)
    head_score: (B,1,Hx,Wx)
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if 'adapter' in yaml_name:
        plt_save = os.path.join(save_path, f'frame{frame_id}_adapter_head_score.png')
    else:
        plt_save = os.path.join(save_path, f'frame{frame_id}_lora_head_score.png')
    if os.path.exists(plt_save):
        print(f"-1")
        return
    bs, _, c = backbone_out.shape
    lens_t = template_feat_size * template_feat_size
    lens_s = backbone_out.shape[1] - lens_t
    search_feat_size = int(math.sqrt(lens_s))
    assert search_feat_size ** 2 == lens_s, "search_feat_size ** 2 must be equal to lens_s"
    backbone_out_z = backbone_out[:, :lens_t].transpose(-1, -2).reshape(bs, -1, template_feat_size, template_feat_size)
    backbone_out_x = backbone_out[:, -lens_s:].transpose(-1, -2).reshape(bs, -1, search_feat_size, search_feat_size)
    backbone_out_z = backbone_out_z.mean(dim=1).squeeze().cpu().numpy()
    backbone_out_x = backbone_out_x.mean(dim=1).squeeze().cpu().numpy()
    head_score_np = head_score.squeeze().cpu().numpy()

    # fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    # fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    # ax = fig.add_subplot(111)
    # ax.imshow(backbone_out_z, cmap='plasma', interpolation='nearest')
    # ax.axis('off')
    # plt.savefig(os.path.join(save_path, f'frame{frame_id}_backbone_out_z.png'))
    #
    # fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    # fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    # ax = fig.add_subplot(111)
    # ax.imshow(backbone_out_x, cmap='plasma', interpolation='nearest')
    # ax.axis('off')
    # plt.savefig(os.path.join(save_path, f'frame{frame_id}_backbone_out_x.png'))

    fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=300)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    ax = fig.add_subplot(111)
    ax.imshow(head_score_np, cmap='plasma', interpolation='nearest')
    ax.axis('off')
    plt.savefig(plt_save)
    plt.close()
