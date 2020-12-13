import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# import coutils
# from coutils import extract_drive_file_id, register_colab_notebooks, \
#                     fix_random_seed, rel_error
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import time
import shutil
import os

def coord_trans(bbox, w_pixel, h_pixel, w_amap=7, h_amap=7, mode='a2p'):
  assert mode in ('p2a', 'a2p'), 'invalid coordinate transformation mode!'
  assert bbox.shape[-1] >= 4, 'the transformation is applied to the first 4 values of dim -1'
  
  if bbox.shape[0] == 0: # corner cases
    return bbox

  resized_bbox = bbox.clone()
  # could still work if the first dim of bbox is not batch size
  # in that case, w_pixel and h_pixel will be scalars
  resized_bbox = resized_bbox.view(bbox.shape[0], -1, bbox.shape[-1])
  invalid_bbox_mask = (resized_bbox == -1) # indicating invalid bbox

  if mode == 'p2a':
    # pixel to activation
    width_ratio = w_pixel * 1. / w_amap
    height_ratio = h_pixel * 1. / h_amap
    resized_bbox[:, :, [0, 2]] /= width_ratio.view(-1, 1, 1)
    resized_bbox[:, :, [1, 3]] /= height_ratio.view(-1, 1, 1)
  else:
    # activation to pixel
    width_ratio = w_pixel * 1. / w_amap
    height_ratio = h_pixel * 1. / h_amap
    resized_bbox[:, :, [0, 2]] *= width_ratio.view(-1, 1, 1)
    resized_bbox[:, :, [1, 3]] *= height_ratio.view(-1, 1, 1)

  resized_bbox.masked_fill_(invalid_bbox_mask, -1)
  resized_bbox.resize_as_(bbox)
  return resized_bbox

def GenerateGrid(batch_size, w_amap=7, h_amap=7, dtype=torch.float32, device='cuda'):
  w_range = torch.arange(0, w_amap, dtype=dtype, device=device) + 0.5
  h_range = torch.arange(0, h_amap, dtype=dtype, device=device) + 0.5

  w_grid_idx = w_range.unsqueeze(0).repeat(h_amap, 1)
  h_grid_idx = h_range.unsqueeze(1).repeat(1, w_amap)
  grid = torch.stack([w_grid_idx, h_grid_idx], dim=-1)
  grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

  return grid


def GenerateAnchor(anc, grid):
  B, H_amap, W_amap, _ = grid.shape
  A, _ = anc.shape
  anchors = torch.zeros(B, A, H_amap, W_amap, 4).to(grid)
  for a in range(A):
    anchors[:, a, :, :, 0] = grid[:, :, :, 0] - anc[a, 0] / 2.0
    anchors[:, a, :, :, 1] = grid[:, :, :, 1] - anc[a, 1] / 2.0
    anchors[:, a, :, :, 2] = grid[:, :, :, 0] + anc[a, 0] / 2.0
    anchors[:, a, :, :, 3] = grid[:, :, :, 1] + anc[a, 1] / 2.0
  return anchors

def GenerateProposal(anchors, offsets, method='YOLO'):
  assert(method in ['YOLO', 'FasterRCNN'])
  proposals = None
  B, A, H, W, _ = anchors.shape
  proposals = torch.zeros_like(anchors)
  proposals_transfer = torch.zeros_like(anchors)
  anchors_transfer = torch.zeros_like(anchors)
  anchors_transfer[:, :, :, :, 0] = (anchors[:, :, :, :, 0] + anchors[:, :, :, :, 2]) / 2
  anchors_transfer[:, :, :, :, 1] = (anchors[:, :, :, :, 1] + anchors[:, :, :, :, 3]) / 2
  anchors_transfer[:, :, :, :, 2] = anchors[:, :, :, :, 2] - anchors[:, :, :, :, 0]
  anchors_transfer[:, :, :, :, 3] = anchors[:, :, :, :, 3] - anchors[:, :, :, :, 1]
  if method == 'YOLO':
    proposals_transfer[:, :, :, :, 0] = anchors_transfer[:, :, :, :, 0] + offsets[:, :, :, :, 0]
    proposals_transfer[:, :, :, :, 1] = anchors_transfer[:, :, :, :, 1] + offsets[:, :, :, :, 1]
    proposals_transfer[:, :, :, :, 2] = anchors_transfer[:, :, :, :, 2] * torch.exp(offsets[:, :, :, :, 2])
    proposals_transfer[:, :, :, :, 3] = anchors_transfer[:, :, :, :, 3] * torch.exp(offsets[:, :, :, :, 3])
  else:
    proposals_transfer[:, :, :, :, 0] = anchors_transfer[:, :, :, :, 0] + offsets[:, :, :, :, 0] * anchors_transfer[:, :, :, :, 2]
    proposals_transfer[:, :, :, :, 1] = anchors_transfer[:, :, :, :, 1] + offsets[:, :, :, :, 1] * anchors_transfer[:, :, :, :, 3]
    proposals_transfer[:, :, :, :, 2] = anchors_transfer[:, :, :, :, 2] * torch.exp(offsets[:, :, :, :, 2])
    proposals_transfer[:, :, :, :, 3] = anchors_transfer[:, :, :, :, 3] * torch.exp(offsets[:, :, :, :, 3])
  proposals[:, :, :, :, 0] = proposals_transfer[:, :, :, :, 0] - proposals_transfer[:, :, :, :, 2] / 2
  proposals[:, :, :, :, 1] = proposals_transfer[:, :, :, :, 1] - proposals_transfer[:, :, :, :, 3] / 2
  proposals[:, :, :, :, 2] = proposals_transfer[:, :, :, :, 0] + proposals_transfer[:, :, :, :, 2] / 2
  proposals[:, :, :, :, 3] = proposals_transfer[:, :, :, :, 1] + proposals_transfer[:, :, :, :, 3] / 2
  return proposals

def IoU(proposals, bboxes):
  iou_mat = None
  B, A, H, W, _ = proposals.shape
  B, N, _ = bboxes.shape
  proposals = proposals.reshape(B, A * H * W, 4).repeat(1, 1, N).reshape(B, A * H * W, N, 4)
  bboxes = bboxes.repeat(1, A * H * W, 1).reshape(B, A * H * W, N, 5)
  xa = torch.max(proposals[:, :, :, 0], bboxes[:, :, :, 0])
  ya = torch.max(proposals[:, :, :, 1], bboxes[:, :, :, 1])
  xb = torch.min(proposals[:, :, :, 2], bboxes[:, :, :, 2])
  yb = torch.min(proposals[:, :, :, 3], bboxes[:, :, :, 3])
  zero = torch.zeros_like(xa)
  intersection = torch.max(zero, (xb - xa)) * torch.max(zero, (yb - ya))
  bbox_area = (bboxes[:, :, :, 2] - bboxes[:, :, :, 0]) * (bboxes[:, :, :, 3] - bboxes[:, :, :, 1])
  proposal_area = (proposals[:, :, :, 2] - proposals[:, :, :, 0]) * (proposals[:, :, :, 3] - proposals[:, :, :, 1])
  union = bbox_area + proposal_area - intersection
  iou_mat = intersection / union
  return iou_mat

def ReferenceOnActivatedAnchors(anchors, bboxes, grid, iou_mat, pos_thresh=0.7, neg_thresh=0.3, method='FasterRCNN'):
  assert(method in ['FasterRCNN', 'YOLO'])

  B, A, h_amap, w_amap, _ = anchors.shape
  N = bboxes.shape[1]

  # activated/positive anchors
  max_iou_per_anc, max_iou_per_anc_ind = iou_mat.max(dim=-1)
  if method == 'FasterRCNN':
    max_iou_per_box = iou_mat.max(dim=1, keepdim=True)[0]
    activated_anc_mask = (iou_mat == max_iou_per_box) & (max_iou_per_box > 0)
    activated_anc_mask |= (iou_mat > pos_thresh) # using the pos_thresh condition as well
    # if an anchor matches multiple GT boxes, choose the box with the largest iou
    activated_anc_mask = activated_anc_mask.max(dim=-1)[0] # Bx(AxH’xW’)
    activated_anc_ind = torch.nonzero(activated_anc_mask.view(-1)).squeeze(-1)

    # GT conf scores
    GT_conf_scores = max_iou_per_anc[activated_anc_mask] # M

    # GT class
    box_cls = bboxes[:, :, 4].view(B, 1, N).expand((B, A*h_amap*w_amap, N))
    GT_class = torch.gather(box_cls, -1, max_iou_per_anc_ind.unsqueeze(-1)).squeeze(-1) # M
    GT_class = GT_class[activated_anc_mask].long()

    bboxes_expand = bboxes[:, :, :4].view(B, 1, N, 4).expand((B, A*h_amap*w_amap, N, 4))
    bboxes = torch.gather(bboxes_expand, -2, max_iou_per_anc_ind.unsqueeze(-1) \
      .unsqueeze(-1).expand(B, A*h_amap*w_amap, 1, 4)).view(-1, 4)
    bboxes = bboxes[activated_anc_ind]
  else:
    bbox_mask = (bboxes[:, :, 0] != -1) # BxN, indicate invalid boxes
    bbox_centers = (bboxes[:, :, 2:4] - bboxes[:, :, :2]) / 2. + bboxes[:, :, :2] # BxNx2

    mah_dist = torch.abs(grid.view(B, -1, 2).unsqueeze(2) - bbox_centers.unsqueeze(1)).sum(dim=-1) # Bx(H'xW')xN
    min_mah_dist = mah_dist.min(dim=1, keepdim=True)[0] # Bx1xN
    grid_mask = (mah_dist == min_mah_dist).unsqueeze(1) # Bx1x(H'xW')xN

    reshaped_iou_mat = iou_mat.view(B, A, -1, N)
    anc_with_largest_iou = reshaped_iou_mat.max(dim=1, keepdim=True)[0] # Bx1x(H’xW’)xN
    anc_mask = (anc_with_largest_iou == reshaped_iou_mat) # BxAx(H’xW’)xN
    activated_anc_mask = (grid_mask & anc_mask).view(B, -1, N)
    activated_anc_mask &= bbox_mask.unsqueeze(1)
    
    # one anchor could match multiple GT boxes
    activated_anc_ind = torch.nonzero(activated_anc_mask.view(-1)).squeeze(-1)
    GT_conf_scores = iou_mat.view(-1)[activated_anc_ind]
    bboxes = bboxes.view(B, 1, N, 5).repeat(1, A*h_amap*w_amap, 1, 1).view(-1, 5)[activated_anc_ind]
    GT_class = bboxes[:, 4].long()
    bboxes = bboxes[:, :4]
    activated_anc_ind = (activated_anc_ind / activated_anc_mask.shape[-1]).long()

#   print('number of pos proposals: ', activated_anc_ind.shape[0])
  activated_anc_coord = anchors.view(-1, 4)[activated_anc_ind]

  # GT offsets
  # bbox and anchor coordinates are x_tl, y_tl, x_br, y_br
  # offsets are t_x, t_y, t_w, t_h
  wh_offsets = torch.log((bboxes[:, 2:4] - bboxes[:, :2]) \
    / (activated_anc_coord[:, 2:4] - activated_anc_coord[:, :2]))

  xy_offsets = (bboxes[:, :2] + bboxes[:, 2:4] - \
    activated_anc_coord[:, :2] - activated_anc_coord[:, 2:4]) / 2.

  if method == "FasterRCNN":
    xy_offsets /= (activated_anc_coord[:, 2:4] - activated_anc_coord[:, :2])
  else:
    assert torch.max(torch.abs(xy_offsets)) <= 0.5, \
      "x and y offsets should be between -0.5 and 0.5! Got {}".format( \
      torch.max(torch.abs(xy_offsets)))

  GT_offsets = torch.cat((xy_offsets, wh_offsets), dim=-1)

  # negative anchors
  negative_anc_mask = (max_iou_per_anc < neg_thresh) # Bx(AxH’xW’)
  negative_anc_ind = torch.nonzero(negative_anc_mask.view(-1)).squeeze(-1)
  negative_anc_ind = negative_anc_ind[torch.randint(0, negative_anc_ind.shape[0], (activated_anc_ind.shape[0],))]
  negative_anc_coord = anchors.view(-1, 4)[negative_anc_ind.view(-1)]
  
  # activated_anc_coord and negative_anc_coord are mainly for visualization purposes
  return activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, \
         activated_anc_coord, negative_anc_coord

def data_visualizer(img, idx_to_class, bbox=None, pred=None):
  img_copy = np.array(img).astype('uint8')

  if bbox is not None:
    for bbox_idx in range(bbox.shape[0]):
      one_bbox = bbox[bbox_idx][:4]
      cv2.rectangle(img_copy, (one_bbox[0], one_bbox[1]), (one_bbox[2],
                  one_bbox[3]), (255, 0, 0), 2)
      if bbox.shape[1] > 4: # if class info provided
        obj_cls = idx_to_class[bbox[bbox_idx][4].item()]
        cv2.putText(img_copy, '%s' % (obj_cls),
                  (one_bbox[0], one_bbox[1]+15),
                  cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

  if pred is not None:
    for bbox_idx in range(pred.shape[0]):
      one_bbox = pred[bbox_idx][:4]
      cv2.rectangle(img_copy, (one_bbox[0], one_bbox[1]), (one_bbox[2],
                  one_bbox[3]), (0, 255, 0), 2)
      
      if pred.shape[1] > 4: # if class and conf score info provided
        obj_cls = idx_to_class[pred[bbox_idx][4].item()]
        conf_score = pred[bbox_idx][5].item()
        cv2.putText(img_copy, '%s, %.2f' % (obj_cls, conf_score),
                    (one_bbox[0], one_bbox[1]+15),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

  return img_copy
  # plt.axis('off')
  # plt.show()