from PIL.Image import Image
import torch
from torch import topk
from torch.nn import functional as F, SmoothL1Loss
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision.ops import nms
from torchvision.transforms import transforms


class COCOObjectDetectionDataset(Dataset):
    """
    Dataset for loading object detection data from COCO format.
    """

    def __init__(self, coco_annotation_file, image_directory, transform=None):
        self.coco = COCO(coco_annotation_file)
        self.image_directory = image_directory
        self.transform = transform
        self.image_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        """returns images and targets for an image at index"""
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = f"{self.image_directory}/{image_info['file_name']}"

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)

        # Get annotations
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        # Extract bounding boxes and labels
        boxes = torch.tensor([ann['bbox'] for ann in annotations], dtype=torch.float32)
        labels = torch.tensor([ann['category_id'] for ann in annotations], dtype=torch.int64)

        # Apply transformations if any
        if self.transform:
            image = self.transform()(image)
            image = torch.Tensor(np.array(image))

        return image, {'boxes': boxes, 'labels': labels}


class MobileNetInspiredDetector(nn.Module):
    """
    Object detection model inspired by MobileNet architecture.
    """

    def __init__(self, num_classes, boxes, input_ch, s=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = boxes

        def depthwise_separable_conv(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.init_conv = nn.Sequential(
            nn.Conv2d(input_ch, 32, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.depthwise_layers = nn.Sequential(
            depthwise_separable_conv(32, 64, stride=1),
            depthwise_separable_conv(64, 128, stride=2),
            depthwise_separable_conv(128, 128, stride=1),
            depthwise_separable_conv(128, 256, stride=2),
            depthwise_separable_conv(256, 256, stride=1),
            depthwise_separable_conv(256, 512, stride=2),
        )
        self.depthwise_deep = nn.Sequential(
            depthwise_separable_conv(512, 512, stride=1),
            depthwise_separable_conv(512, 512, stride=1),
            depthwise_separable_conv(512, 512, stride=1),
            depthwise_separable_conv(512, 512, stride=1),
            depthwise_separable_conv(512, 1024, stride=2)
        )
        self.extras = nn.Sequential(
            depthwise_separable_conv(1024, 512, stride=2),
            depthwise_separable_conv(512, 256, stride=2),
            depthwise_separable_conv(256, 128, stride=1),
        )
        self.loc_layers = nn.Sequential(
            # nn.Conv2d(512, self.num_boxes * 4, kernel_size=3, padding=1),  # For 6 default boxes per feature map
            # nn.Conv2d(512, self.num_boxes * 4, kernel_size=3, padding=1),
            # nn.Conv2d(512, self.num_boxes * 4, kernel_size=3, padding=1),
            # nn.Conv2d(512, self.num_boxes * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, self.num_boxes * 2, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_boxes * 2, kernel_size=3, padding=1),
            nn.Conv2d(128, self.num_boxes*2, kernel_size=3, padding=1)
        )
        self.conf_layers = nn.ModuleList([
            # nn.Conv2d(512, self.num_boxes * self.num_classes, kernel_size=3, padding=1),
            # nn.Conv2d(512, self.num_boxes * self.num_classes, kernel_size=3, padding=1),
            # nn.Conv2d(512, self.num_boxes * self.num_classes, kernel_size=3, padding=1),
            # nn.Conv2d(512, self.num_boxes * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, self.num_boxes * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_boxes * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(128, self.num_boxes * num_classes, kernel_size=3, padding=1)
        ])


    def forward(self, x):
        x = x.permute(2, 0, 1).contiguous()
        x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        x = self.init_conv(x)
        # print(f"Initial convolution was boomed.")
        # print(x.shape)
        features = []
        x = self.depthwise_layers(x)
        x = self.depthwise_deep(x)
        # for layer in self.depthwise_deep:
        #     x = layer(x)
        #     features.append(x)
        # print(f"Depthwise Convolution was boomed.")
        for layer in self.extras:
            x = layer(x)
            features.append(x)
        locs = []
        confs = []

        for i, feature in enumerate(features):
            locs.append(self.loc_layers[i](feature).permute(0, 2, 3, 1).contiguous())
            confs.append(self.conf_layers[i](feature).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([loc.view(loc.size(0), -1, 2) for loc in locs], 1)
        conf = torch.cat([conf.view(conf.size(0), -1, self.num_classes) for conf in confs], 1)
        conf = F.sigmoid(conf)
        # print("The shape of loc and conf are ", loc.shape, conf.shape)
        return loc, conf


# class BoundingBoxProcessor(nn.Module):
#     """
#     Module for processing bounding box predictions and calculating losses.
#     """
#
#     def __init__(self, num_f_classes, batches, ):
#         super().__init__()
#         self.num_classes = num_f_classes
#         self.locations, self.confidence = 0, 0
#         # self.targets = targets
#         self.loc1_boxes = {}
#         self.loc2_boxes = {}
#         self.confidence_1 = {}
#         self.confidence_2 = {}
#         self.label_1 = {}
#         self.label_2 = {}
#         self.confidences = []
#         self.boxes = []
#         self.label = []
#
#     def process_predictions(self, conf_score, predictions, targets, device, start_batch, end_batch, iou_threshold=0.1):
#         # total_loss = 0
#         pred_locs = torch.zeros(size = (targets.shape[0],targets.shape[1],targets.shape[2]),device = device)
#         self.locations, self.confidence = predictions
#         self.start_batch = start_batch
#         self.end_batch = end_batch
#         self.batches = end_batch - start_batch + 1
#         self.loss_tensor = torch.zeros((self.batches, 1), device=device)
#         total_loss = torch.zeros(1, device=device, requires_grad=True)
#         print(f"The total number of batches are: {self.batches}")
#         for batch in range(self.start_batch, self.end_batch):
#
#             print("Batch:", batch)
#             conf = self.confidence[batch]
#             loc = self.locations[batch]
#             # print(f"Confidence has a shape of {conf.shape}")
#             mask_1 = conf[:, 0] > conf_score
#             if not mask_1.any():
#                 print(f"No predictions above confidence threshold in batch {batch}")
#                 continue  # Skip to the next batch
#
#             valid_locbox_class1 = loc[mask_1]
#             valid_cobox_class1 = conf[mask_1][:, 0]
#             x_1, y_1 = valid_locbox_class1[:, 0], valid_locbox_class1[:, 1]
#             h_1, w_1 = valid_locbox_class1[:, 2] + y_1, valid_locbox_class1[:, 3] + x_1
#             boxes_class1 = torch.stack([x_1, y_1, h_1, w_1], dim=1)
#             if boxes_class1.numel() == 0:
#                 print(f"No valid boxes in batch {batch}")
#                 continue  # Skip to the next batch
#             nms_indices_class1 = nms(boxes_class1, valid_cobox_class1, iou_threshold)
#             self.loc1_boxes[batch] = boxes_class1[nms_indices_class1]
#             self.confidence_1[batch] = valid_cobox_class1[nms_indices_class1]
#             self.label_1 = (self.confidence_1[batch] >= conf_score).float()
#             item_locs = self.loc1_boxes[batch]
#             item_confs = self.label_1
#             target_boxes = targets['boxes']
#             target_labels = targets['labels']
#
#             if item_locs.numel() == 0 or target_boxes.numel() == 0:
#                 print(f"No predictions or targets in batch {batch}")
#                 continue
#             ious = box_iou(item_locs, target_boxes.to(device))
#             if ious.numel() == 0:
#                 print(f"No valid IoUs in batch {batch}")
#                 continue
#             top_n = target_boxes.shape[0]
#             top_ious, top_indices = ious.topk(top_n, dim=0)
#             best_iou, best_idx = ious.max(dim=0)
#             best_pred_loc = item_locs[top_indices, :]
#             top_item_confs = item_confs[top_indices]
#             valid_mask = torch.zeros(len(item_locs), dtype=torch.bool)
#             valid_mask[best_idx] = True
#             loc_feed = best_pred_loc[0].float()
#             loc_target_feed = target_boxes.float().to(device)
#             conf_feed = top_item_confs[0].float()
#             conf_target_feed = target_labels.flatten().float().to(device)
#             pred_locs[:, 0] = loc_feed[:, 0] * loc_target_feed[:, 0]
#             pred_locs[:, 1] = loc_feed[:, 1] * loc_target_feed[:, 1]
#             pred_locs[:, 2] = loc_feed[:, 0] * loc_target_feed[:, 2]
#             pred_locs[:, 3] = loc_feed[:, 1] * loc_target_feed[:, 3]
#             loc_loss = F.smooth_l1_loss(pred_locs, loc_target_feed, reduction='sum')
#             conf_loss = F.cross_entropy(conf_feed, conf_target_feed, reduction='sum')
#
#             num_matched = valid_mask.sum().float()
#             total_loss += (loc_loss + conf_loss) / num_matched
#             # self.loss_tensor[batch] = total_loss
#         self.boxes = self.loc1_boxes
#         self.confidences = self.confidence_1
#         self.label = self.label_1
#
#         return self.boxes, self.confidences, self.label, total_loss / self.batches


def box_iou(boxes1, boxes2):
    """
    Compute Intersection over Union (IoU) between two sets of boxes.

    Args:
    boxes1, boxes2: Tensors of shape (N, 4) and (M, 4) respectively,
                    where N and M are the number of boxes,
                    and each box is represented as (x1, y1, x2, y2)

    Returns:
    Tensor of shape (N, M) containing pairwise IoU values
    """
    boxes2 = boxes2
    # print("Shaes are", boxes1.shape, "and", boxes2.shape)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def data_loader(ann_file, img_file, init_batch, end_batch):
    # coco = COCO("ticks.coco/train/_annotations.coco.json")
    Images = []
    T = []
    X_ = []
    Aobj = COCOObjectDetectionDataset(coco_annotation_file=ann_file, image_directory=img_file,
                                      transform=transforms.ToPILImage)
    for i in range(init_batch, end_batch):
        x, tgt = Aobj.__getitem__(i)
        X_.append(x)
        T.append(tgt)
    Images = [X_, T]
    X = Images
    return X_, T, init_batch, end_batch


# class ImprovedBoundingBoxProcessor(nn.Module):
#     def __init__(self, num_classes, alpha=0.25, gamma=2.0):
#         super().__init__()
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.gamma = gamma
#         self.criterion = F.smooth_l1_loss()
#
#
#     def forward(self, predictions, targets, conf_threshold=0.6, iou_threshold=0.5):
#         locations, confidences = predictions
#         batch_size = len(locations)
#         total_loss = 0
#
#         for batch in range(batch_size):
#             batch_loss = self.process_single_batch(
#                 locations[batch], confidences[batch],
#                 targets[batch]['boxes'], targets[batch]['labels'],
#                 conf_threshold, iou_threshold
#             )
#             total_loss += batch_loss
#
#         return total_loss / batch_size
#
#     def process_single_batch(self, loc, conf, target_boxes, target_labels, conf_threshold, iou_threshold):
#         device = loc.device
#         target_boxes = target_boxes.to(device)
#
#         # Ensure conf is of shape (num_predictions, num_classes)
#         if conf.dim() == 3:
#             conf = conf.squeeze(0)
#
#         confidence_scores, predicted_classes = conf.max(dim=1)
#         mask = confidence_scores > conf_threshold
#
#         if not mask.any():
#             print("Not found any mask.")
#             return torch.tensor(0.0, device=device)
#
#         # Apply mask to the second dimension of loc
#         filtered_loc = loc[:, mask, :]  # This preserves the batch dimension if present
#         filtered_conf = conf[mask]
#         filtered_scores = confidence_scores[mask]
#
#         if filtered_loc.numel() == 0 or target_boxes.numel() == 0:
#             print("No locations found")
#             return torch.tensor(0.0, device=device)
#
#         # Squeeze the batch dimension if it exists
#         filtered_loc = filtered_loc.squeeze(0)
#         ious = box_iou(filtered_loc, target_boxes)
#         max_ious, max_indices = ious.max(dim=1)
#         score = filtered_conf[:,0]
#         # print("Shape of conf is ",score,filtered_conf.squeeze.shape)
#         nms_boxes = nms(filtered_loc,score,iou_threshold)
#         # positive_mask = max_ious > iou_threshold
#         if not nms_boxes.any():
#             print("No mask.")
#             return torch.tensor(0.0, device=device)
#         print(f"The shapes are {filtered_loc.shape}, {filtered_conf.shape},{nms_boxes.shape}")
#         matched_loc = filtered_loc[nms_boxes,:]
#         matched_conf = filtered_conf[nms_boxes,:]
#         matched_target_boxes = target_boxes
#         matched_target_labels = target_labels
#
#         loc_loss = F.smooth_l1_loss(matched_loc, matched_target_boxes, reduction='sum')
#         conf_loss = self.focal_loss(matched_conf, matched_target_labels)
#
#         num_positives = nms_boxes.sum().float()
#         total_loss = (loc_loss + conf_loss) / num_positives
#         print("Reached here.")
#         return total_loss
#
#         # ... rest of the function remains the same
#     # def process_single_batch(self, loc, conf, target_boxes, target_labels, conf_threshold, iou_threshold):
#     #     device = loc.device
#     #     confidence_scores, predicted_classes = conf.max(dim=1)
#     #     mask = confidence_scores > conf_threshold
#     #
#     #     if not mask.any():
#     #         return torch.tensor(0.0, device=device)
#     #
#     #     filtered_loc = loc[mask]
#     #     filtered_conf = conf[mask]
#     #     filtered_scores = confidence_scores[mask]
#     #
#     #     if filtered_loc.numel() == 0 or target_boxes.numel() == 0:
#     #         return torch.tensor(0.0, device=device)
#
#
#
#     def focal_loss(self, pred, target):
#         ce_loss = F.cross_entropy(pred, target, reduction='none')
#         p_t = torch.exp(-ce_loss)
#         loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
#         return loss.sum()
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------
class ImprovedBoundingBoxProcessor2(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.criterion = SmoothL1Loss(reduction='sum')


    def loss_forward(self, predictions, targets, conf_threshold=0.5, iou_threshold=0.5):
        locations, confidences = predictions
        # batch_size = len(locations)
        total_loss = 0

        batch_loss = self.process_single_batch(
            locations, confidences,
            targets['boxes'], targets['labels'],
            conf_threshold, iou_threshold
        )
        print("Batch loss: ", batch_loss)
        total_loss += batch_loss

        return total_loss

    def process_single_batch(self, loc, conf, target_boxes, target_labels, conf_threshold, iou_threshold):
        device = loc.device
        target_boxes = target_boxes.to(device)
        target_labels = target_labels.to(device)

        if conf.dim() == 3:
            conf = conf.squeeze(0)


        confidence_scores, predicted_classes = conf.max(dim=1)


        mask = confidence_scores > conf_threshold

        if not mask.any():
            # mask
            print("Not found any mask.")
            return torch.tensor(0.001, device=device,requires_grad=True)

        # Apply mask to the second dimension of loc
        filtered_loc = loc[:, mask, :]  # This preserves the batch dimension if present
        filtered_conf = conf[mask]
        filtered_scores = confidence_scores[mask]


        if filtered_loc.numel() == 0 or target_boxes.numel() == 0:
            print("No locations found")
            return torch.tensor(0.001, device=device,requires_grad=True)



        tgtbox_expanded = target_boxes.unsqueeze(2).expand(-1, target_boxes.size(1), filtered_loc.size(1), 4)  # (1, 3, 5, 4)
        predloc_expanded = filtered_loc.unsqueeze(1).expand(-1, target_boxes.size(1), filtered_loc.size(1), 2)  # (1, 3, 5, 2)
        loc_determined = torch.zeros(size=(target_boxes.size(0), target_boxes.size(1) * filtered_loc.size(1), target_boxes.size(2)),device=device)
        loc_determined[:, :, 0] = (tgtbox_expanded[:, :, :, 0] * predloc_expanded[:, :, :, 0]).view(1, -1)
        loc_determined[:, :, 1] = (tgtbox_expanded[:, :, :, 1] * predloc_expanded[:, :, :, 1]).view(1, -1)
        loc_determined[:, :, 2] = (tgtbox_expanded[:, :, :, 2] * predloc_expanded[:, :, :, 0]).view(1, -1)
        loc_determined[:, :, 3] = (tgtbox_expanded[:, :, :, 3] * predloc_expanded[:, :, :, 1]).view(1, -1)

        score = filtered_conf[:,0]

        pred_loc_final = loc_determined.squeeze(0)
        nms_boxes = nms(pred_loc_final,score,iou_threshold)
        # positive_mask = max_ious > iou_threshold
        if not nms_boxes.any():
            print("No mask.")
            return torch.tensor(0.001, device=device,requires_grad=True)

        matched_loc = pred_loc_final[nms_boxes,:]
        matched_conf = filtered_conf[nms_boxes,:]
        k = target_labels.shape[0]
        matched_conf,indices= topk(matched_conf,k,dim=0)
        matched_conf = (matched_conf[:,0]> 0.5).to(torch.float64)

        matched_loc = matched_loc[indices,:]
        matched_target_boxes = target_boxes
        matched_target_labels = target_labels.to(torch.float64)
        # pred_locs[:, 0] = matched_loc[:, 0] * matched_target_boxes[:, 0]
        # pred_locs[:, 1] = matched_loc[:, 1] * matched_target_boxes[:, 1]
        # pred_locs[:, 2] = matched_loc[:, 0] * matched_target_boxes[:, 2]
        # pred_locs[:, 3] = matched_loc[:, 1] * matched_target_boxes[:, 3]
        # pred_box = matched_loc
        loc_loss = self.criterion(matched_loc, matched_target_boxes,)
        conf_loss = self.focal_loss(matched_conf, matched_target_labels)


        num_positives = nms_boxes.sum().float()
        total_loss = (loc_loss + conf_loss) / num_positives
        print("Reached here.")
        del score,matched_loc,target_boxes,target_labels,
        del num_positives,loc_loss,conf_loss,matched_conf,matched_target_boxes,matched_target_labels,filtered_conf,filtered_scores,filtered_loc,conf_threshold,iou_threshold
        return total_loss
    def focal_loss(self, pred, target):

        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return loss.sum()




#-----------------------------------------------------------------
#-----------------------------------------------------------------------------------
# class ImprovedBoundingBoxProcessor2(nn.Module):
#     def __init__(self, num_classes, alpha=0.25, gamma=2.0):
#         super().__init__()
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.gamma = gamma
#         self.criterion = SmoothL1Loss(reduction='sum')
#
#
#     def loss_forward(self, predictions, targets, conf_threshold=0.6, iou_threshold=0.5):
#         locations, confidences = predictions
#         # batch_size = len(locations)
#         total_loss = 0
#
#         batch_loss = self.process_single_batch(
#             locations, confidences,
#             targets['boxes'], targets['labels'],
#             conf_threshold, iou_threshold
#         )
#         print("Batch loss: ", batch_loss)
#         total_loss += batch_loss
#
#         return total_loss
#
#     def process_single_batch(self, loc, conf, target_boxes, target_labels, conf_threshold, iou_threshold):
#         device = loc.device
#         target_boxes = target_boxes.to(device)
#         target_labels = target_labels.to(device)
#         # pred_locs = torch.zeros(size = (target_boxes.size(0), target_boxes.size(1),target_boxes.size(2)), device = device)
#         # Ensure conf is of shape (num_predictions, num_classes)
#         if conf.dim() == 3:
#             conf = conf.squeeze(0)
#
#         # confidence_scores, predicted_classes = conf.max(dim=1)
#         # print(f"Shape before applying max: {conf.shape}")
#         confidence_scores, predicted_classes = conf.max(dim=1)
#         print(f"Shape after applying max: {confidence_scores.shape}, {predicted_classes.shape}")
#         print("The shape of locations and confidenc is ",loc.shape,conf.shape)
#         mask = confidence_scores > conf_threshold
#
#         if not mask.any():
#             # mask
#             print("Not found any mask.")
#             return torch.tensor(0.001, device=device,requires_grad=True)
#
#         # Apply mask to the second dimension of loc
#         filtered_loc = loc[:, mask, :]  # This preserves the batch dimension if present
#         filtered_conf = conf[mask]
#         filtered_scores = confidence_scores[mask]
#         print(f"The shape of filtered score and conf is {filtered_conf.shape},{filtered_scores.shape}")
#
#         if filtered_loc.numel() == 0 or target_boxes.numel() == 0:
#             print("No locations found")
#             return torch.tensor(0.001, device=device,requires_grad=True)
#
#         # Squeeze the batch dimension if it exists
#         print(f"The shape of filtered_loc and target_boxes is {filtered_loc.shape},{target_boxes.shape}")
#
#
#         tgtbox_expanded = target_boxes.unsqueeze(2).expand(-1, target_boxes.size(1), filtered_loc.size(1), 4)  # (1, 3, 5, 4)
#         predloc_expanded = filtered_loc.unsqueeze(1).expand(-1, target_boxes.size(1), filtered_loc.size(1), 2)  # (1, 3, 5, 2)
#         loc_determined = torch.zeros(size=(target_boxes.size(0), target_boxes.size(1) * filtered_loc.size(1), target_boxes.size(2)),device=device)
#         loc_determined[:, :, 0] = (tgtbox_expanded[:, :, :, 0] * predloc_expanded[:, :, :, 0]).view(1, -1)
#         loc_determined[:, :, 1] = (tgtbox_expanded[:, :, :, 1] * predloc_expanded[:, :, :, 1]).view(1, -1)
#         loc_determined[:, :, 2] = (tgtbox_expanded[:, :, :, 2] * predloc_expanded[:, :, :, 0]).view(1, -1)
#         loc_determined[:, :, 3] = (tgtbox_expanded[:, :, :, 3] * predloc_expanded[:, :, :, 1]).view(1, -1)
#         # ious = box_iou(filtered_loc, target_boxes)
#         # max_ious, max_indices = ious.max(dim=1)
#         score = filtered_conf[:,0]
#         # print("Shape of conf is ",score,filtered_conf.squeeze.shape)
#         pred_loc_final = loc_determined.squeeze(0)
#         nms_boxes = nms(pred_loc_final,score,iou_threshold)
#         # positive_mask = max_ious > iou_threshold
#         if not nms_boxes.any():
#             print("No mask.")
#             return torch.tensor(0.001, device=device,requires_grad=True)
#
#         matched_loc = pred_loc_final[nms_boxes,:]
#         matched_conf = filtered_conf[nms_boxes,:]
#         k = target_labels.shape[0]
#         matched_conf,indices= topk(matched_conf,k,dim=0)
#         matched_conf = (matched_conf[:,0]> 0.5).to(torch.float64)
#
#         matched_loc = matched_loc[indices,:]
#         matched_target_boxes = target_boxes
#         matched_target_labels = target_labels.to(torch.float64)
#         # pred_locs[:, 0] = matched_loc[:, 0] * matched_target_boxes[:, 0]
#         # pred_locs[:, 1] = matched_loc[:, 1] * matched_target_boxes[:, 1]
#         # pred_locs[:, 2] = matched_loc[:, 0] * matched_target_boxes[:, 2]
#         # pred_locs[:, 3] = matched_loc[:, 1] * matched_target_boxes[:, 3]
#         # pred_box = matched_loc
#         loc_loss = self.criterion(matched_loc, matched_target_boxes,)
#         conf_loss = self.focal_loss(matched_conf, matched_target_labels)
#
#
#         num_positives = nms_boxes.sum().float()
#         total_loss = (loc_loss + conf_loss) / num_positives
#         print("Reached here.")
#         del score,matched_loc,target_boxes,target_labels,
#         del num_positives,loc_loss,conf_loss,matched_conf,matched_target_boxes,matched_target_labels,filtered_conf,filtered_scores,filtered_loc,conf_threshold,iou_threshold
#         return total_loss
#     def focal_loss(self, pred, target):
#         # print(f"The values of the target boxes are: {target.dtype} and prediction are: {pred}")
#         ce_loss = F.cross_entropy(pred, target, reduction='none')
#         p_t = torch.exp(-ce_loss)
#         loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
#         return loss.sum()

