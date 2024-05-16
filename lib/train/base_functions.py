import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import LasherTir, LSOTB
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.train_print_interval = cfg.TRAIN.TRAIN_PRINT_INTERVAL
    settings.val_print_interval = cfg.TRAIN.VAL_PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.fix_bn = getattr(cfg.TRAIN, "FIX_BN", True)  # add for fixing base model bn layer
    settings.ffd_output_layer = getattr(cfg.PROMPT, "REFOCUS_FFD_OUTPUT_LAYER", None)  # None 5


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASHERTIR_train", "LASHERTIR_test", "LSOTB_train", "LSOTB_val"]
        if name == "LASHERTIR_train":
            datasets.append(LasherTir(settings.env.lasher_dir, image_loader=image_loader, split='train'))
        if name == "LASHERTIR_test":
            datasets.append(LasherTir(settings.env.lasher_dir, image_loader=image_loader, split='test'))
        if name == "LSOTB_train":
            datasets.append(LSOTB(settings.env.lsotb_dir, image_loader=image_loader, split='train'))
        if name == "LSOTB_val":
            datasets.append(LSOTB(settings.env.lsotb_dir, image_loader=image_loader, split='val'))

    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    data_processing_val = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    print("sampler_mode", sampler_mode)
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders
    if cfg.DATA.VAL.DATASETS_NAME[0] is None:
        loader_val = None
    else:
        dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                              p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                              samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                              max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                              num_template_frames=settings.num_template, processing=data_processing_val,
                                              frame_sample_mode=sampler_mode, train_cls=train_cls)
        val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
        loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                               num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                               epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    prompt_type = getattr(cfg.PROMPT, "TYPE", "")
    if prompt_type == "FFT" or prompt_type == "no_prompt":
        if prompt_type == "FFT":
            print("Full fine-tuning on downstream tasks. Training all model parameters.")
        else:
            print("Training normally as OSTrack.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
    elif prompt_type == "headprobe":
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "box_head" in n and p.requires_grad]}
        ]
        for n, p in net.named_parameters():
            if "box_head" not in n:
                p.requires_grad = False

    elif prompt_type == "LoRA":
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "lora" in n and p.requires_grad]}
        ]
        for n, p in net.named_parameters():
            if "lora" not in n:
                p.requires_grad = False
    elif prompt_type == "VPT":
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "vpt" in n and p.requires_grad]}
        ]
        for n, p in net.named_parameters():
            if "vpt" not in n:
                p.requires_grad = False

    elif prompt_type == "Adapter":
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "adapter" in n and p.requires_grad]}
        ]
        for n, p in net.named_parameters():
            if "adapter" not in n:
                p.requires_grad = False

    elif "ReFocus" in prompt_type:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "refocus" in n and p.requires_grad]}
        ]
        for n, p in net.named_parameters():
            if "refocus" not in n:
                p.requires_grad = False
    else:
        raise ValueError(f"Do not support {prompt_type} now!")

    if is_main_process():
        print("Learnable parameters are shown below.")
        for n, p in net.named_parameters():
            if p.requires_grad:
                print(n)

        n_parameters = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print(f'Number of trainable params: {n_parameters / (1e6)}M')

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
