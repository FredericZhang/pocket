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

class InteractionHead(nn.Module):
    """
    Interaction head that constructs and classifies box pairs based on object detections

    Arguments:

    [REQUIRES ARGS]
        box_pair_pooler(BoxPairMultiScaleRoIAlign): The module that applies a mask on the union of 
            a box pair and proceed to extract pooled features based on torchvision.ops.MultiScaleRoIAlign
        pooler_output_shape(tuple): (C, H, W)
        representation_size(int): Size of the intermediate representation
        num_classes(int): Number of output classes
        object_class_to_target_class(list[Tensor]): Each element in the list maps an object class
            to corresponding target classes

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
            box_pair_pooler,
            pooler_output_shape, 
            representation_size, 
            num_classes,
            object_class_to_target_class,
            fg_iou_thresh=0.5, num_box_pairs_per_image=512, positive_fraction=0.25,
            box_score_thresh=0.2, box_nms_thresh=0.5):
        
        super().__init__()

        self.box_pair_pooler = box_pair_pooler
        self.box_pair_head = nn.Sequential(
            nn.Linear(torch.as_tensor(pooler_output_shape).prod(), representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, representation_size),
            nn.ReLU()
        )
        self.box_pair_logistic = nn.Linear(representation_size, num_classes)

        self.num_classes = num_classes  

        self.object_class_to_target_class = object_class_to_target_class

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
        for detection in detections:
            boxes, scores, labels = detection.values()

            # Remove low scoring boxes
            keep_idx = torch.nonzero(scores > self.box_score_thresh).squeeze(1)
            boxes = boxes[keep_idx, :].view(-1, 4)
            scores = scores[keep_idx].view(-1)
            labels = labels[keep_idx].view(-1)

            # Class-wise non-maximum suppresion
            keep_idx = box_ops.batched_nms(boxes, scores, labels, self.box_nms_thresh)
            boxes = boxes[keep_idx, :].view(-1, 4)
            scores = scores[keep_idx].view(-1)
            labels = labels[keep_idx].view(-1)

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

        for idx, (h_idx, o_idx) in enumerate(paired_idx):
            mapped_scores[idx, self.object_class_to_target_class[labels[o_idx]]] = \
                scores[h_idx] * scores[o_idx]
        
        return mapped_scores

    def append_ground_truth_box_pairs(self, boxes_h, boxes_o, prior_scores, targets):
        """
        Arguments:
            boxes_h(Tensor[N, 4])
            boxes_o(Tensor[N, 4])
            prior_scores(Tensor[N, K])
            targets(dict[Tensor]): {
                "boxes_h": Tensor[G, 4],
                "boxes_o": Tensor[G, 4],
                "object": Tensor[G]
            }
        Returns:
            boxes_h(Tensor[N+G, 4])
            ...
        """
        boxes_h = torch.cat([boxes_h, targets['boxes_h']], 0)
        boxes_o = torch.cat([boxes_o, targets['boxes_o']], 0)

        gt_scores = torch.zeros(len(targets['boxes_h']), self.num_classes)
        for idx, obj_cls in enumerate(targets['object']):
            gt_scores[idx, self.object_class_to_target_class[obj_cls]] = 1
        prior_scores = torch.cat([prior_scores, gt_scores], 0)

        return boxes_h, boxes_o, prior_scores

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
                neg_idx[torch.randperm(len(neg_idx))[:num_neg_to_sample]],
            ])
        else:
            num_pos_to_sample = int(self.num_box_pairs_per_image * self.positive_fraction)
            num_neg_to_sample = self.num_box_pairs_per_image - num_pos_to_sample
            return torch.cat([
                pos_idx[torch.randperm(len(pos_idx))[:num_pos_to_sample]],
                neg_idx[torch.randperm(len(neg_idx))[:num_neg_to_sample]],
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
        all_interactions = []
        all_prior_scores = []
        for idx in range(len(boxes)):
            # Find detections of human instances
            h_idx = torch.nonzero(labels[idx] == 49).squeeze(1)
            paired_idx = torch.cat([
                v.flatten()[:, None] for v in torch.meshgrid(
                    h_idx, torch.arange(len(labels[idx])))
            ], 1)

            # Remove pairs of the same human instance
            keep_idx = (paired_idx[:, 0] != paired_idx[:, 1]).nonzero().squeeze(1)
            paired_idx = paired_idx[keep_idx, :].view(-1, 2)

            # Construct box pairs
            boxes_h = boxes[idx][paired_idx[:, 0]].view(-1, 4)
            boxes_o = boxes[idx][paired_idx[:, 1]].view(-1, 4)
            interactions = None
            prior_scores = self.map_object_scores_to_interaction_scores(
                scores[idx], labels[idx], paired_idx)

            # Assign labels to constructed box pairs
            if self.training:
                target_in_image = targets[idx]
                boxes_h, boxes_o, prior_scores = self.append_ground_truth_box_pairs(
                    boxes_h, boxes_o, prior_scores, target_in_image)

                interactions = torch.zeros(len(boxes_h), self.num_classes)

                fg_match = torch.nonzero(torch.min(
                    box_ops.box_iou(boxes_h, target_in_image['boxes_h']),
                    box_ops.box_iou(boxes_o, target_in_image['boxes_o'])
                ) >= self.fg_iou_thresh).view(-1, 2)
                interactions[
                    fg_match[:, 0], 
                    target_in_image['hoi'][fg_match[:, 1]]
                ] = 1

                # Subsample up to a specified number of box pairs 
                # with fixed positive-negative ratio
                sampled_idx = self.subsample(interactions)
                boxes_h = boxes_h[sampled_idx].view(-1, 4)
                boxes_o = boxes_o[sampled_idx].view(-1, 4)
                interactions = interactions[sampled_idx].view(-1, self.num_classes)
                prior_scores = prior_scores[sampled_idx].view(-1, self.num_classes)

            all_boxes_h.append(boxes_h)
            all_boxes_o.append(boxes_o)
            all_interactions.append(interactions)
            all_prior_scores.append(prior_scores)

        return all_boxes_h, all_boxes_o, all_interactions, all_prior_scores


    def compute_interaction_classification_loss(self, class_logits, prior_scores, box_pair_labels):
        # Ignore interaction classes with zero prior scores
        keep_idx = prior_scores.nonzero()
        interaction_scores = prior_scores[keep_idx[:, 0], keep_idx[:, 1]] * \
            torch.sigmoid(class_logits)[keep_idx[:, 0], keep_idx[:, 1]]
        labels = box_pair_labels[keep_idx[:, 0], keep_idx[:, 1]]

        return torch.nn.functional.binary_cross_entropy(interaction_scores, labels)

    def postprocess(self, class_logits, prior_scores, boxes_h, boxes_o):
        num_boxes = [len(boxes_per_image) for boxes_per_image in boxes_h]
        interaction_scores = (_cat(prior_scores)
                * torch.sigmoid(class_logits)).split(num_boxes)

        results = []
        for scores, b_h, b_o in zip(interaction_scores, boxes_h, boxes_o):
            keep_idx = scores.nonzero()
            results.append(dict(
                boxes_h = b_h[keep_idx[:, 0]].view(-1, 4),
                boxes_o = b_o[keep_idx[:, 0]].view(-1, 4),
                labels = keep_idx[:, 1].view(-1),
                scores = scores[keep_idx[:, 0], keep_idx[:, 1]].view(-1),
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
            results(list[dict]): During evaluation, return a dict of detected interacitons
                "boxes_h": Tensor[M, 4]
                "boxes_o": Tensor[M, 4]
                "labels": Tensor[M]
                "scores": Tensor[M]
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"
        else:
            self.filter_detections(detections)

        box_coords = [detection['boxes'] for detection in detections]
        box_labels = [detection['labels'] for detection in detections]
        box_scores = [detection['scores'] for detection in detections]

        boxes_h, boxes_o, box_pair_labels, prior_scores = self.pair_up_boxes_and_assign_to_targets(
            box_coords, box_labels, box_scores, targets)

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
