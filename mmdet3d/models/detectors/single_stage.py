# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from ...structures.det3d_data_sample import OptSampleList, SampleList
from .base import Base3DDetector


@MODELS.register_module()
class SingleStage3DDetector(Base3DDetector):
    """SingleStage3DDetector.

    This class serves as a base class for single-stage 3D detectors.
    This updated version includes an optional `net_loss` for auxiliary
    supervision on intermediate network features.

    Args:
        backbone (dict): Config dict of detector's backbone.
        neck (dict, optional): Config dict of neck. Defaults to None.
        bbox_head (dict, optional): Config dict of box head. Defaults to None.
        net_loss (dict, optional): Config dict of network structure loss.
            Defaults to None.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.
        init_cfg (dict or ConfigDict, optional): The config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 net_loss: OptConfigType = None,  # Added net_loss
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        # Initialize net_loss if provided
        if net_loss is not None:
            self.net_loss = MODELS.build(net_loss)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_net_loss(self) -> bool:
        """bool: Whether the detector has a loss related to structure."""
        return hasattr(self, 'net_loss') and self.net_loss is not None

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data Samples.

        Returns:
            dict: A dictionary of loss components.
        """
        features, net_info = self.extract_feat(batch_inputs_dict)
        losses = dict()

        # Calculate net_loss from intermediate features if applicable
        if self.with_net_loss and net_info is not None:
            losses_net = self.net_loss(net_info)
            losses.update(losses_net)

        # Calculate losses from the bounding box head
        bbox_losses = self.bbox_head.loss(features, batch_data_samples, **kwargs)
        losses.update(bbox_losses)
        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data Samples.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the input samples.
        """
        # Ignore net_info during inference
        features, _ = self.extract_feat(batch_inputs_dict)
        results_list = self.bbox_head.predict(features, batch_data_samples, **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions

    def _forward(self,
                 batch_inputs_dict: dict,
                 data_samples: OptSampleList = None,
                 **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process without any post-processing.

         Args:
            batch_inputs_dict (dict): The model input dict.
            data_samples (List[:obj:`Det3DDataSample`]): The Data Samples.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        # Ignore net_info during forward pass for non-training modes
        features, _ = self.extract_feat(batch_inputs_dict)
        results = self.bbox_head.forward(features)
        return results

    def extract_feat(
        self, batch_inputs_dict: Dict[str, Tensor]
    ) -> Union[Tuple[torch.Tensor], Dict[str, Tensor]]:
        """Directly extract features from the backbone+neck.

        This method now returns features and an optional `net_info` dictionary
        for calculating `net_loss`.

        Args:
            batch_inputs_dict (dict): The model input dict.

        Returns:
            Tuple[Union[Tuple[Tensor], Dict], Union[Dict, None]]:
                A tuple containing the features and an optional dictionary
                of intermediate network info.
        """
        points = batch_inputs_dict['points']
        stack_points = torch.stack(points)
        
        net_info = None
        x = self.backbone(stack_points)
        
        # Check if backbone returns features and net_info
        if self.with_net_loss and isinstance(x, (list, tuple)):
            x, net_info = x

        if self.with_neck:
            x = self.neck(x)
            # Check if neck returns features and net_info
            if self.with_net_loss and isinstance(x, (list, tuple)):
                x, net_info = x

        return x, net_info