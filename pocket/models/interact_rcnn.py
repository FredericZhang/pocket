"""
Implementation of Interact R-CNN

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torchvision.ops.boxes as box_ops
from torch import nn
from torchvision.ops._utils import _cat
from torchvision.models.detection import transform

from .faster_rcnn import fasterrcnn_resnet_fpn
from ..ops import sinkhorn_knopp_norm2d
from ..ops import SimpleBoxPairPool, MaskedBoxPairPool

class InteractionHead(nn.Module):
    """
    Interaction head that constructs and classifies box pairs based on object detections

    Arguments:

    [REQUIRES ARGS]
        box_pair_pooler(nn.Module): Module that pools a feature vector for a given box pair
        pooler_output_shape(tuple): (C, H, W)
        representation_size(int): Size of the intermediate representation
        num_classes(int): Number of output classes
        object_class_to_target_class(list[Tensor]): Each element in the list maps an object class
            to corresponding target classes
        human_idx(int): The index of human/person class in all objects

    [OPTIONAL ARGS]
        fg_iou_thresh(float): Minimum intersection over union between proposed box pairs and ground
            truth box pairs to be considered as positive
        num_box_pairs_per_image(int): Number of box pairs used in training for each image
        positive_fraction(float): The propotion of positive box pairs used in training
        box_score_thresh(float): The box score threshold used to filter object detections 
            during evaluation
        box_nms_thresh(float): NMS threshold to filter object detections during evaluation
    """
    def __init__(self,
            # Architectural parameters
            box_pair_pooler,
            pooler_output_shape, 
            representation_size, 
            num_classes,
            # Parameters for box pair construction
            object_class_to_target_class, human_idx,
            # Parameters for training
            fg_iou_thresh=0.5, num_box_pairs_per_image=512, positive_fraction=0.25,
            # Parameters for inference
            box_score_thresh=0.2, box_nms_thresh=0.5):
        
        super().__init__()

        self.box_pair_pooler = box_pair_pooler

        input_size = torch.as_tensor(pooler_output_shape).prod().item()
        self.box_pair_head = nn.Sequential(
            nn.Linear(input_size, representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, representation_size),
            nn.ReLU()
        )
        self.box_pair_logistic = nn.Linear(representation_size, num_classes)

        self.num_classes = num_classes  

        self.object_class_to_target_class = object_class_to_target_class
        self.human_idx = human_idx

        self.fg_iou_thresh = fg_iou_thresh
        self.num_box_pairs_per_image = num_box_pairs_per_image
        self.positive_fraction = positive_fraction
        # Parameters used to filter box detections during evaluation
        self.box_score_thresh = box_score_thresh
        self.box_nms_thresh = box_nms_thresh

    def filter_detections(self, detections):
        """
        detections(list[dict]): Object detections with following keys 
            "boxes": Tensor[N,4]
            "labels": Tensor[N] 
            "scores": Tensor[N]
        """
        results = []
        for detection in detections:
            boxes = detection['boxes']
            labels = detection['labels']
            scores = detection['scores']

            # Remove low scoring boxes
            keep_idx = torch.nonzero(scores > self.box_score_thresh).squeeze(1)
            boxes = boxes[keep_idx].view(-1, 4)
            scores = scores[keep_idx].view(-1)
            labels = labels[keep_idx].view(-1)

            # Class-wise non-maximum suppresion
            keep_idx = box_ops.batched_nms(
                boxes, scores, labels, 
                self.box_nms_thresh
            )
            boxes = boxes[keep_idx].view(-1, 4)
            scores = scores[keep_idx].view(-1)
            labels = labels[keep_idx].view(-1)

            results.append(dict(boxes=boxes, labels=labels, scores=scores))

        return results

    def map_object_scores_to_interaction_scores(self, scores, labels, paired_idx):
        """
        Arguments:
            scores(Tensor[N]): Object confidence scores
            labels(Tensor[N]): Object labels of HICO80 with zero-based index
            paired_idx(Tensor[M, 2])
        Returns:
            mapped_scores(Tensor[M, K]): Object confidence scores mapped to interaction classes
                that contain the corresponding object
        """
        mapped_scores = torch.zeros(len(paired_idx), self.num_classes,
            dtype=scores.dtype, device=scores.device)

        h_idx, o_idx = paired_idx.unbind(1)
        # Product of object detection scores for each pair
        prod = scores[h_idx] * scores[o_idx]

        obj_cls = labels[o_idx]
        # Find mapped HOI indices for each pair
        hoi_idx = [self.object_class_to_target_class[obj] 
            for obj in obj_cls]
        # Duplicate box pair indices for each HOI class
        pair_idx = [i for i, hois in enumerate(hoi_idx) for _ in range(len(hois))]
        # Flatten mapped HOI indices
        flat_hoi_idx = [hoi for hois in hoi_idx for hoi in hois]

        mapped_scores[pair_idx, flat_hoi_idx] = prod[pair_idx]
        
        return mapped_scores

    def append_ground_truth_box_pairs(self, paired_idx, boxes, labels, scores, targets):
        """
        Arguments:
            paired_idx(Tensor[M, 2])
            boxes(Tensor[N, 4])
            labels(Tensor[N])
            scores(Tensor[N])
            targets(dict[Tensor]): {
                "boxes_h": Tensor[G, 4],
                "boxes_o": Tensor[G, 4],
                "object": Tensor[G]
            }
        Returns:
            paired_idx(Tensor[M+G, 4])
            boxes(Tensor[N+2G, 4])
            labels(Tensor[N+2G])
            scores(Tensor[N+2G])
        """
        num_gt = len(targets['boxes_h'])
        paired_idx = torch.cat([
            torch.arange(
                2 * num_gt,
                dtype=paired_idx.dtype,
                device=paired_idx.device
            ).view(2, -1).transpose(0, 1),
            paired_idx
        ], 0)
        boxes = torch.cat([
            targets['boxes_h'],
            targets['boxes_o'],
            boxes
        ], 0)
        labels = torch.cat([
            self.human_idx * torch.ones(
                num_gt,
                dtype=labels.dtype,
                device=labels.device
            ), targets['object'], labels
        ], 0)
        # FIXME Assign ones as object detection scores to GT
        scores = torch.cat([
            torch.ones(
                num_gt * 2,
                dtype=scores.dtype,
                device=scores.device
            ), scores
        ], 0)

        return paired_idx, boxes, labels, scores

    def subsample(self, labels):
        """
        Arguments:
            labels(Tensor[N, K]): Binary labels
        Returns:
            Tensor[M]
        """
        is_positive = labels.sum(1)
        pos_idx = torch.nonzero(is_positive).squeeze(1)
        neg_idx = torch.nonzero(is_positive == 0).squeeze(1)

        # If there is a lack of positives, use all and keep the ratio
        if len(pos_idx) < self.num_box_pairs_per_image * self.positive_fraction:
            num_neg_to_sample = int(len(pos_idx) 
                * (1 - self.positive_fraction) / self.positive_fraction)
            return torch.cat([
                pos_idx,
                neg_idx[torch.randperm(len(neg_idx),
                    device=labels.device)[:num_neg_to_sample]],
            ])
        else:
            num_pos_to_sample = int(self.num_box_pairs_per_image * self.positive_fraction)
            num_neg_to_sample = self.num_box_pairs_per_image - num_pos_to_sample
            return torch.cat([
                pos_idx[torch.randperm(len(pos_idx),
                    device=labels.device)[:num_pos_to_sample]],
                neg_idx[torch.randperm(len(neg_idx),
                    device=labels.device)[:num_neg_to_sample]],
            ])

    def pair_up_boxes_and_assign_to_targets(self, boxes, labels, scores, targets=None):
        """
        Arguments:
            boxes(list[Tensor[N, 4]])
            labels(list[Tensor[N]]): Object labels of HICO80 with zero-based index
            scores(list[Tensor[N]]): Object confidence scores
            targets(dict[Tensor]): {
                "boxes_h": Tensor[G, 4],
                "boxes_o": Tensor[G, 4],
                "hoi": Tensor[G]
                "object": Tensor[G]
            }
        Returns:
            all_boxes_h(list[Tensor[M, 4]])
            all_boxes_o(list[Tensor[M, 4]])
            all_interactions(list[Tensor[M, K]]): Binary labels for each box pair
            all_prior_scores(list[Tensor[M, K]]): Product of human and object box scores 
                mapped to interaction classes
        """
        if self.training and targets is None:
            raise AssertionError("Targets should be passed during training")
        
        all_boxes_h = []
        all_boxes_o = []
        all_labels = []
        all_prior_scores = []
        for idx, (boxes_in_image, labels_in_image, scores_in_images) in enumerate(
            zip(boxes, labels, scores)
        ):
            # Find detections of human instances
            h_idx = torch.nonzero(labels_in_image == self.human_idx).squeeze(1)
            paired_idx = torch.cat([
                v.flatten()[:, None] for v in torch.meshgrid(
                    h_idx, torch.arange(len(labels_in_image), device=h_idx.device))
            ], 1)

            # Remove pairs of the same human instance
            keep_idx = (paired_idx[:, 0] != paired_idx[:, 1]).nonzero().squeeze(1)
            paired_idx = paired_idx[keep_idx, :].view(-1, 2)

            # Placeholders
            boxes_h = boxes_o = hoi_labels = None

            # Assign labels to constructed box pairs and perform subsampling
            if self.training:
                targets_in_image = targets[idx]
                paired_idx, boxes_in_image, labels_in_image, scores_in_images = \
                    self.append_ground_truth_box_pairs(
                        paired_idx,
                        boxes_in_image,
                        labels_in_image,
                        scores_in_images,
                        targets_in_image
                    )

                hoi_labels = torch.zeros(len(paired_idx), self.num_classes,
                        device=boxes_in_image.device)
                boxes_h = boxes_in_image[paired_idx[:, 0]].view(-1, 4)
                boxes_o = boxes_in_image[paired_idx[:, 1]].view(-1, 4)

                fg_match = torch.nonzero(torch.min(
                    box_ops.box_iou(boxes_h, targets_in_image['boxes_h']),
                    box_ops.box_iou(boxes_o, targets_in_image['boxes_o'])
                ) >= self.fg_iou_thresh).view(-1, 2)
                hoi_labels[
                    fg_match[:, 0], 
                    targets_in_image['hoi'][fg_match[:, 1]]
                ] = 1

                # Subsample up to a specified number of box pairs 
                # with fixed positive-negative ratio
                sampled_idx = self.subsample(hoi_labels)
                paired_idx = paired_idx[sampled_idx].view(-1, 2)
                boxes_h = boxes_h[sampled_idx].view(-1, 4)
                boxes_o = boxes_o[sampled_idx].view(-1, 4)
                hoi_labels = hoi_labels[sampled_idx].view(-1, self.num_classes)

            if boxes_h is None or boxes_o is None:
                boxes_h = boxes_in_image[paired_idx[:, 0]]
                boxes_o = boxes_in_image[paired_idx[:, 1]]
            prior_scores = self.map_object_scores_to_interaction_scores(
                scores_in_images, labels_in_image, paired_idx
            )

            all_boxes_h.append(boxes_h)
            all_boxes_o.append(boxes_o)
            all_labels.append(hoi_labels)
            all_prior_scores.append(prior_scores)

        return all_boxes_h, all_boxes_o, all_labels, all_prior_scores


    def compute_interaction_classification_loss(self, 
            class_logits, prior_scores, box_pair_labels):
        # Ignore interaction classes with zero prior scores
        i, j = prior_scores.nonzero().unbind(1)

        interaction_scores = prior_scores[i, j] * \
            torch.sigmoid(class_logits)[i, j]
        labels = box_pair_labels[i, j]

        return torch.nn.functional.binary_cross_entropy(interaction_scores, labels)

    def sinkhorn_knopp_normalisation(self, boxes_h, boxes_o, scores):
        """
        Arguments:
            boxes_h(Tensor[M, 4])
            boxes_o(Tensor[M, 4])
            scores(Tensor[M, C])
        Returns:
            Tensor[M, C]
        """
        unique_h, h_intra_idx = boxes_h.unique(dim=0, return_inverse=True)
        unique_o, o_intra_idx = boxes_o.unique(dim=0, return_inverse=True)

        scores_ = torch.zeros(
            len(unique_h), len(unique_o), scores.shape[-1],
            device=scores.device, dtype=scores.dtype
        )
        scores_[h_intra_idx, o_intra_idx, :] = scores
        del scores

        for i in range(scores_.shape[-1]):
            scores_[:, :, i], _ = sinkhorn_knopp_norm2d(scores_[:, :, i])

        return scores_[h_intra_idx, o_intra_idx, :]

    def postprocess(self, class_logits, prior_scores, boxes_h, boxes_o):
        num_boxes = [len(boxes_per_image) for boxes_per_image in boxes_h]
        interaction_scores = (_cat(prior_scores)
                * torch.sigmoid(class_logits)).split(num_boxes)

        results = []
        for scores, b_h, b_o in zip(interaction_scores, boxes_h, boxes_o):
            scores *= self.sinkhorn_knopp_normalisation(
                b_h, b_o, scores)

            keep_cls = [s.nonzero().squeeze(1) for s in scores]
            keep_box = torch.as_tensor([bool(len(pred_cls)) for pred_cls in keep_cls],
                device=keep_cls[0].device)

            results.append(dict(
                boxes_h=b_h[keep_box].view(-1, 4),
                boxes_o=b_o[keep_box].view(-1, 4),
                labels=keep_cls,
                scores=[scores[i, pred_cls] for i, pred_cls in enumerate(keep_cls)]
            ))

        return results

    def forward(self, features, detections, targets=None):
        """
        Arguments:
            features(list[Tensor]): Image pyramid with each tensor corresponding to
                a feature level
            detections(list[dict]): Object detections with following keys 
                boxes(Tensor[N,4]),
                labels(Tensor[N]) 
                scores(Tensor[N])
            targets(list[dict]): Interaction targets with the following keys
                boxes_h(Tensor[N, 4])
                boxes_o(Tensor[N, 4])
                hoi(Tensor[N])
                object(Tensor[N])
                verb(Tensor[N])
        Returns:
            loss(dict): During training, return a dict that contains the interaction loss
            results(list[dict]): During evaluation, return dicts of detected interacitons
                "boxes_h": Tensor[M, 4]
                "boxes_o": Tensor[M, 4]
                "labels": list(Tensor)
                "scores": list(Tensor)
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"
        else:
            detections = self.filter_detections(detections)

        box_coords = [detection['boxes'] for detection in detections]
        box_labels = [detection['labels'] for detection in detections]
        box_scores = [detection['scores'] for detection in detections]

        boxes_h, boxes_o, box_pair_labels, prior_scores = self.pair_up_boxes_and_assign_to_targets(
            box_coords, box_labels, box_scores, targets)

        # No box pairs available
        # This happens when there are no detected persons
        if sum([len(b) for b in boxes_h]) == 0:
            return None

        box_pair_features = self.box_pair_pooler(features, boxes_h, boxes_o)
        box_pair_features = box_pair_features.flatten(start_dim=1)
        box_pair_features = self.box_pair_head(box_pair_features)
        class_logits = self.box_pair_logistic(box_pair_features)

        if self.training:
            loss = dict(interaction_loss=self.compute_interaction_classification_loss(
                class_logits, _cat(prior_scores), _cat(box_pair_labels)
            ))
            return loss
        
        results = self.postprocess(
            class_logits, prior_scores, boxes_h, boxes_o)

        return results

class InteractRCNNTransform(transform.GeneralizedRCNNTransform):
    """
    Transformations for input image and target (box pairs)

    Arguments(Positional):
        min_size(int)
        max_size(int)
        image_mean(list[float] or tuple[float])
        image_std(list[float] or tuple[float])

    Refer to torchvision.models.detection for more details
    """
    def __init__(self, *args):
        super().__init__(*args)

    def resize(self, image, target):
        """
        Override method to resize box pairs
        """
        h, w = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        scale_factor = min(
            self.min_size[0] / min_size,
            self.max_size / max_size
        )

        image = nn.functional.interpolate(image[None], 
            scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]
        if target is None:
            return image, target

        target['boxes_h'] = transform.resize_boxes(target['boxes_h'],
            (h, w), image.shape[-2:])
        target['boxes_o'] = transform.resize_boxes(target['boxes_o'],
            (h, w), image.shape[-2:])

        return image, target

    def postprocess(self, results, image_shapes, original_image_sizes):
        if self.training:
            return results

        for pred, im_s, o_im_s in zip(results, image_shapes, original_image_sizes):
            boxes_h, boxes_o = pred['boxes_h'], pred['boxes_o']
            boxes_h = transform.resize_boxes(boxes_h, im_s, o_im_s)
            boxes_o = transform.resize_boxes(boxes_o, im_s, o_im_s)
            pred['boxes_h'], pred['boxes_o'] = boxes_h, boxes_o

        return results

class TrainableHead(nn.Module):
    """
    Add backbone to make interaction head trainable

    Arguments:
        cls_corr(list[Tensor]): One-to-many mapping from object classes to interaction classes
        human_idx(int): The index of human/person class in all objects
        backbone(str): The name of backbone CNN to be used. 
            Refer to torchvision.models.deteciton.backbone_utils for more details.
        ...
    """
    def __init__(self, cls_corr, human_idx,
            # Backbone parameters
            backbone='resnet50', pretrained=True,
            # Transformation parameters
            min_size=800, max_size=1333, 
            image_mean=None, image_std=None,
            # Pooler parameters
            output_size=7, spatial_scale=None, sampling_ratio=2, 
            # Parameters for masked pooling
            masked_pool=True, mem_limit=8, reserve=128,
            # MLP parameters
            representation_size=1024, num_classes=600,
            **kwargs):
        super().__init__()

        self.backbone = fasterrcnn_resnet_fpn(backbone, pretrained=pretrained).backbone
    
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = InteractRCNNTransform(min_size, max_size, image_mean, image_std)

        if spatial_scale is None:
            spatial_scale = [1/4, 1/8, 1/16, 1/32]
        if masked_pool:
            pooler = MaskedBoxPairPool(
                output_size=output_size,
                spatial_scale=spatial_scale,
                sampling_ratio=sampling_ratio,
                mem_limit=mem_limit,
                reserve=reserve
            )
        else:
            pooler = SimpleBoxPairPool(
                output_size=output_size,
                spatial_scale=spatial_scale,
                sampling_ratio=sampling_ratio
            )
        
        self.interaction_head = InteractionHead(
            pooler,
            (self.backbone.out_channels, output_size, output_size),
            representation_size, num_classes,
            cls_corr, human_idx, **kwargs
        )

    def preprocess(self, images, detections, targets=None):
        self.original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)

        for det, o_im_s, im_s in zip(detections, self.original_image_sizes, images.image_sizes):
            boxes = det['boxes']
            boxes = transform.resize_boxes(boxes, o_im_s, im_s)
            det['boxes'] = boxes

        return images, detections, targets

    def forward(self, images, detections, targets=None):
        """
        Arguments:
            images(list[Tensor])
            detections(list[dict])
            targets(list[dict])
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images, detections, targets = self.preprocess(
                images, detections, targets)

        with torch.no_grad():
            features = self.backbone(images.tensors)
        # Remove the last max pooled features in fpn
        features = [v for v in features.values()]
        features = features[:-1]

        results = self.interaction_head(features, detections, targets)

        if results is None:
            return results

        return self.transform.postprocess(results, images.image_sizes, self.original_image_sizes)

    """Override methods to only train interaction head"""
    def parameters(self):
        return self.interaction_head.parameters()
    def state_dict(self):
        return self.interaction_head.state_dict()
    def load_state_dict(self, state_dict):
        self.interaction_head.load_state_dict(state_dict)


class InteractRCNN(nn.Module):
    def __init__(self, backbone, rpn, roi_heads, interaction_heads, transform):
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.interaction_heads = interaction_heads
        self.transform = transform

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]], optional): ground-truth boxes present in the image
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals,
            images.image_sizes, targets)
        detections, interaction_loss = self.interaction_heads(features, detections,
            images.image_sizes, targets)    
        detections = self.transform.postprocess(detections,
            images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(interaction_loss)

        if self.training:
            return losses

        return detections
