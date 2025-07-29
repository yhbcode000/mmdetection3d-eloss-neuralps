# model settings
_base_ = [
    './xnet_single-SECOND-3class.py',
    ]

model = dict(
    pts_backbone=dict(
        _delete_=True,
        type='SECOND_INFO',
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    
    # 图像特征提取 Image Feature Extraction
    img_backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(1, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[128, 512],
        out_channels=512,
        num_outs=2),

    # 特征级别融合网络 Feature Fusion
    pre_fusion=dict(
        _delete_=True,
        type='PreFusionCat',
        img_channels=512,
        img_out_channels=512,
        pts_channels=512,
        pts_out_channels=512,
        img_levels=2,
        pts_levels=1,
        dropout_ratio=0.2), 
    get_graph=None,
    fusion=dict(
        _delete_=True,
        type='FusionNN',
        in_channels=512+512,
        out_channels=512),
    fusion_neck=dict(
        _delete_=True,
        type='FusionNeckNN',
        in_channels=512*3,
        out_channels=512),
    
    net_loss=dict(
        _delete_=True,
        type = 'EntropyLoss'),
    
    pts_bbox_head=dict(
        feat_channels=512)
)
