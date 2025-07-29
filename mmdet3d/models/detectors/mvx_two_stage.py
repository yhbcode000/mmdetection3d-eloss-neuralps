# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from .base import Base3DDetector


@MODELS.register_module()
class MVXTwoStageDetector(Base3DDetector):
    """Base class of Multi-modality VoxelNet.

    Args:
        pts_voxel_encoder (dict, optional): Point voxelization
            encoder layer. Defaults to None.
        pts_middle_encoder (dict, optional): Middle encoder layer
            of points cloud modality. Defaults to None.
        pts_fusion_layer (dict, optional): Fusion layer.
            Defaults to None.
        img_backbone (dict, optional): Backbone of extracting
            images feature. Defaults to None.
        pts_backbone (dict, optional): Backbone of extracting
            points features. Defaults to None.
        img_neck (dict, optional): Neck of extracting
            image features. Defaults to None.
        pts_neck (dict, optional): Neck of extracting
            points features. Defaults to None.
        net_loss (dict, optional): Config dict of network structure loss.
            Defaults to None.
        pts_bbox_head (dict, optional): Bboxes head of
            point cloud modality. Defaults to None.
        img_roi_head (dict, optional): RoI head of image
            modality. Defaults to None.
        img_rpn_head (dict, optional): RPN head of image
            modality. Defaults to None.
        train_cfg (dict, optional): Train config of model.
            Defaults to None.
        test_cfg (dict, optional): Train config of model.
            Defaults to None.
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`Det3DDataPreprocessor`. Defaults to None.
    """

    def __init__(self,
                 pts_voxel_encoder: Optional[dict] = None,
                 pts_middle_encoder: Optional[dict] = None,
                 pts_fusion_layer: Optional[dict] = None,
                 img_backbone: Optional[dict] = None,
                 pts_backbone: Optional[dict] = None,
                 img_neck: Optional[dict] = None,
                 pts_neck: Optional[dict] = None,
                 net_loss: Optional[dict] = None,
                 pts_bbox_head: Optional[dict] = None,
                 img_roi_head: Optional[dict] = None,
                 img_rpn_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 **kwargs):
        super(MVXTwoStageDetector, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor, **kwargs)

        if pts_voxel_encoder:
            self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = MODELS.build(pts_backbone)
        if pts_fusion_layer:
            self.pts_fusion_layer = MODELS.build(pts_fusion_layer)
        if pts_neck is not None:
            self.pts_neck = MODELS.build(pts_neck)
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = MODELS.build(pts_bbox_head)

        if img_backbone:
            self.img_backbone = MODELS.build(img_backbone)
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
        if img_rpn_head is not None:
            self.img_rpn_head = MODELS.build(img_rpn_head)
        if img_roi_head is not None:
            self.img_roi_head = MODELS.build(img_roi_head)

        # Initialize net_loss if provided
        if net_loss is not None:
            self.net_loss = MODELS.build(net_loss)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_img_shared_head(self):
        """bool: Whether the detector has a shared head in image branch."""
        return hasattr(self,
                       'img_shared_head') and self.img_shared_head is not None

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self,
                       'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return hasattr(self,
                       'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_fusion(self):
        """bool: Whether the detector has a fusion layer."""
        return hasattr(self,
                       'pts_fusion_layer') and self.fusion_layer is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None

    @property
    def with_voxel_encoder(self):
        """bool: Whether the detector has a voxel encoder."""
        return hasattr(self,
                       'voxel_encoder') and self.voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        """bool: Whether the detector has a middle encoder."""
        return hasattr(self,
                       'middle_encoder') and self.middle_encoder is not None
    @property
    def with_net_loss(self) -> bool:
        """bool: Whether the detector has a loss related to structure."""
        return hasattr(self, 'net_loss') and self.net_loss is not None
    
    def _forward(self):
        pass

    def extract_img_feat(self, img: Tensor, input_metas: List[dict]) -> dict:
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in input_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(
        self,
        voxel_dict: Dict[str, Tensor],
        points: Optional[List[Tensor]] = None,
        img_feats: Optional[Sequence[Tensor]] = None,
        batch_input_metas: Optional[List[dict]] = None
    ) -> Tuple[Sequence[Tensor], Optional[dict]]:
        """Extract features of points and optional network info.

        Returns:
            Tuple[Sequence[tensor], dict]: A tuple contains points features
            and an optional dict of network info for loss calculation.
        """
        if not self.with_pts_bbox:
            return None, None
        
        net_info = None
        voxel_features = self.pts_voxel_encoder(voxel_dict['voxels'],
                                                voxel_dict['num_points'],
                                                voxel_dict['coors'], img_feats,
                                                batch_input_metas)
        batch_size = voxel_dict['coors'][-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, voxel_dict['coors'],
                                     batch_size)
        
        x = self.pts_backbone(x)
        # Check if backbone returns features and net_info
        if self.with_net_loss and isinstance(x, (list, tuple)):
            x, net_info = x

        if self.with_pts_neck:
            x = self.pts_neck(x)
            # Check if neck also returns features and net_info
            if self.with_net_loss and isinstance(x, (list, tuple)):
                x, net_info = x
                
        return x, net_info

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_input_metas: List[dict]) -> tuple:
        """Extract features from images and points.

        Returns:
            tuple: A tuple contains image features, point cloud features,
            and an optional dict of network info.
        """
        voxel_dict = batch_inputs_dict.get('voxels', None)
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        
        img_feats = self.extract_img_feat(imgs, batch_input_metas)
        
        pts_feats, net_info = self.extract_pts_feat(
            voxel_dict,
            points=points,
            img_feats=img_feats,
            batch_input_metas=batch_input_metas)
        
        return (img_feats, pts_feats, net_info)

    def loss(self, batch_inputs_dict: Dict[List, torch.Tensor],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> Dict[str, Tensor]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data Samples.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img_feats, pts_feats, net_info = self.extract_feat(batch_inputs_dict,
                                                     batch_input_metas)
        losses = dict()

        # Calculate net_loss from intermediate features if applicable
        if self.with_net_loss and net_info is not None:
            losses_net = self.net_loss(net_info)
            losses.update(losses_net)

        if pts_feats:
            losses_pts = self.pts_bbox_head.loss(pts_feats, batch_data_samples,
                                                 **kwargs)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.loss_imgs(img_feats, batch_data_samples)
            losses.update(losses_img)
            
        return losses

    def loss_imgs(self, x: List[Tensor],
                  batch_data_samples: List[Det3DDataSample], **kwargs):
        """Forward function for image branch.

        This function works similar to the forward function of Faster R-CNN.

        Args:
            x (list[torch.Tensor]): Image features of shape (B, C, H, W)
                of multiple levels.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict: Losses of each branch.
        """
        losses = dict()
        # RPN forward and loss
        if self.with_img_rpn:
            proposal_cfg = self.test_cfg.rpn
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)
            rpn_losses, rpn_results_list = self.img_rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg, **kwargs)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in keys:
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)

        else:
            if 'proposals' in batch_data_samples[0]:
                # use pre-defined proposals in InstanceData
                # for the second stage
                # to extract ROI features.
                rpn_results_list = [
                    data_sample.proposals for data_sample in batch_data_samples
                ]
            else:
                rpn_results_list = None
        # bbox head forward and loss
        if self.with_img_bbox:
            roi_losses = self.img_roi_head.loss(x, rpn_results_list,
                                                batch_data_samples, **kwargs)
            losses.update(roi_losses)
        return losses

    def predict_imgs(self,
                     x: List[Tensor],
                     batch_data_samples: List[Det3DDataSample],
                     rescale: bool = True,
                     **kwargs) -> InstanceData:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            x (List[Tensor]): Image features from FPN.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.
        """

        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.img_rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        results_list = self.img_roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale, **kwargs)
        return results_list

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.
        Args:
            batch_inputs_dict (dict): The model input dict.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data Samples.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the input sample.
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        
        # Ignore net_info during inference
        img_feats, pts_feats, _ = self.extract_feat(batch_inputs_dict,
                                                  batch_input_metas)
                                                  
        if pts_feats and self.with_pts_bbox:
            results_list_3d = self.pts_bbox_head.predict(
                pts_feats, batch_data_samples, **kwargs)
        else:
            results_list_3d = None

        if img_feats and self.with_img_bbox:
            results_list_2d = self.predict_imgs(img_feats, batch_data_samples,
                                                **kwargs)
        else:
            results_list_2d = None

        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 results_list_3d,
                                                 results_list_2d)
        return detsamples
