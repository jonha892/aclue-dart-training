### Loss
import torch
import torchvision.ops as ops
import torch.nn as nn
import torch.nn.functional as F


def match_boxes(predicted_boxes, gt_boxes, iou_threshold=0.0):
    """
    Match anchor boxes to ground truth boxes based on IoU criteria.

    Args:
        predicted_boxes (Tensor): Anchor boxes with shape (N, 4), where N is the number of anchor boxes.
        gt_boxes (Tensor): Ground truth boxes with shape (M, 4), where M is the number of ground truth boxes.
        iou_threshold (float): IoU threshold for matching anchor boxes to ground truth boxes.

    Returns:
        matched_idxs (Tensor): Indices of ground truth boxes with most overlap, with shape (N).
        positive_mask (Tensor): Mask indicating positive iou threshold, with shape (N,).
    """
    num_anchors = predicted_boxes.size(0)
    num_gt_boxes = gt_boxes.size(0)

    if num_anchors == 0 or num_gt_boxes == 0:
        raise ValueError("No anchor boxes or ground truth boxes")

    #print(f"shape of anchor_boxes: {predicted_boxes.shape}")
    #print(f"shape of gt_boxes: {gt_boxes.shape}")
    #print(f"predicted_boxes: {predicted_boxes}")
    #print(f"gt_boxes: {gt_boxes}")
    # Todo find candidate for predicted box or gt box? which way works better?
    iou_matrix = ops.box_iou(predicted_boxes, gt_boxes)
    #print("iou_matrix", iou_matrix)

    max_iou_values, matched_idxs = iou_matrix.max(dim=1)
    positive_mask = max_iou_values >= iou_threshold

    #print(f"shape of iou_matrix: {iou_matrix.shape}")
    #print(f"shape of matched_idxs: {matched_idxs.shape}")
    #print(f"shape of positive_mask: {positive_mask.shape}")

    return matched_idxs, positive_mask


class PushkinLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(PushkinLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, ground_truth):
        predicted_boxes, predicted_classes = predictions
        gt_boxes, gt_classes = ground_truth

        #print(f"shape of predicted_classes: {predicted_classes.shape}")
        #print(f"shape of predicted_boxes: {predicted_boxes.shape}")

        #batch_matched_idxs, batch_positive_masks = [], []

        batch_size = predicted_boxes.size(0)
        batch_losses = []
        #print(f"batch_size: {batch_size}")
        for i in range(batch_size):
            predicted_boxes_i = predicted_boxes[i]
            gt_boxes_i = gt_boxes[i]
            predicted_classes_i = predicted_classes[i]
            gt_classes_i = gt_classes[i]
            matched_idxs, positive_mask = match_boxes(
                predicted_boxes_i, gt_boxes_i
            )
            #print(f"positive mask: {positive_mask}")
            #print(f"matched_idxs: {matched_idxs}")
            
            if (positive_mask == False).all():
                continue
            

            matched_predicted_boxes = predicted_boxes_i[positive_mask]
            matched_predicted_classes = predicted_classes_i[positive_mask]

            matched_gt_boxes = gt_boxes_i[matched_idxs]
            matched_gt_boxes = matched_gt_boxes[positive_mask]
            matched_gt_classes = gt_classes_i[matched_idxs]
            matched_gt_classes = matched_gt_classes[positive_mask]

            #print(f"shape of matched_predicted_boxes: {matched_predicted_boxes.shape}")
            #print(f"shape of matched_predicted_classes: {matched_predicted_classes.shape}")
            #print(f"shape of matched_gt_boxes: {matched_gt_boxes.shape}")
            #print(f"shape of matched_gt_classes: {matched_gt_classes.shape}")

            l = self.loss(matched_predicted_boxes, matched_predicted_classes, matched_gt_boxes, matched_gt_classes)
            batch_losses.append(l)

            #print("end iteration", i)

        if len(batch_losses) == 0:
            print("batch_losses is empty")
            return torch.tensor(0.0, requires_grad=True)
        return sum(batch_losses)
    
    def loss(self, matched_box_preds, matched_class_preds, box_gt, class_gt):
        # Localization loss (Smooth L1 Loss)
        #print("matched_box_preds", matched_box_preds)
        #print("box_gt", box_gt)
        loc_loss = F.smooth_l1_loss(matched_box_preds, box_gt)

        # Classification loss (Focal Loss)
        #cls_loss = self.focal_loss(matched_class_preds, class_gt)

        # Total loss
        loss = loc_loss * 10# + torch.tensor(0)
        #print(loss)
        return loss

    def focal_loss(self, preds, labels):
        # Compute one-hot encoded targets
        one_hot = F.one_hot(labels, num_classes=preds.size(-1)).float()

        # Compute probabilities and weights for focal loss
        probs = torch.sigmoid(preds)
        pt = torch.where(one_hot == 1, probs, 1 - probs)
        alpha = torch.where(one_hot == 1, self.alpha, 1 - self.alpha)

        # Compute focal loss
        loss = -alpha * (1 - pt) ** self.gamma * torch.log(pt)

        # Sum the loss along the class dimension
        cls_loss = loss.sum(dim=-1)

        return cls_loss.mean()