from timm.models.layers import to_2tuple
from lib.models.ostrack.vit import vit_base_patch16_224, vit_small_patch16_224, vit_tiny_patch16_224
from lib.models.ostrack.vit_noshare import vit_small_patch16_base_patch16_224, vit_tiny_patch16_base_patch16_224

backbone_dict = {
    # share backbone
    'vit_small_patch16_224': vit_small_patch16_224,
    'vit_base_patch16_224': vit_base_patch16_224,
    'vit_tiny_patch16_224': vit_tiny_patch16_224,
    # do not share backbone
    'vit_small_patch16_base_patch16_224': vit_small_patch16_base_patch16_224,
    'vit_tiny_patch16_base_patch16_224': vit_tiny_patch16_base_patch16_224
}

def build_two_same_backbone(pretrained, cfg):
    assert cfg.MODEL.BACKBONE.TYPE in backbone_dict, "Not support such backbone type"
    backbone = backbone_dict[cfg.MODEL.BACKBONE.TYPE](pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                        template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                        prompt_type=cfg.PROMPT.TYPE,
                                        vpt_prompt_tokens=cfg.PROMPT.NUM_TOKEN,
                                        lora_r=cfg.PROMPT.LORA_R,
                                        lora_alpha=cfg.PROMPT.LORA_ALPHA,
                                        lora_dropout=cfg.PROMPT.LORA_DROPOUT,
                                        td_num=cfg.PROMPT.REFOCUS_TD_NUM
                                        )

    return backbone

def build_not_shared_backbone(pretrained, cfg):
    backbone = backbone_dict[cfg.MODEL.BACKBONE.TYPE](pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                        template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                        prompt_type=cfg.PROMPT.TYPE,
                                        lora_r=cfg.PROMPT.LORA_R,
                                        td_num=cfg.PROMPT.REFOCUS_TD_NUM
                                        )
    return backbone