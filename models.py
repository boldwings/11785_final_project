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
# import cv2
import copy
import time
import shutil
import os
from util import *

class ProposalModule(nn.Module):
  def __init__(self, in_dim, hidden_dim=256, num_anchors=9, drop_ratio=0.3):
    super().__init__()

    assert(num_anchors != 0)
    self.num_anchors = num_anchors
    self.base = nn.Sequential(
        nn.Conv2d(in_dim, hidden_dim, 3, 1, 1),
        nn.Dropout(p=drop_ratio),
        nn.LeakyReLU(),
        nn.Conv2d(hidden_dim, 6 * self.num_anchors, 1, 1, 0)
    )

  def _extract_anchor_data(self, anchor_data, anchor_idx):
    B, A, D, H, W = anchor_data.shape
    anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
    extracted_anchors = anchor_data[anchor_idx]
    return extracted_anchors

  def forward(self, features, pos_anchor_coord=None, \
              pos_anchor_idx=None, neg_anchor_idx=None):
    if pos_anchor_coord is None or pos_anchor_idx is None or neg_anchor_idx is None:
      mode = 'eval'
    else:
      mode = 'train'
    conf_scores, offsets, proposals = None, None, None
    predictions = self.base(features)
    B, _, H, W = predictions.shape
    conf_scores = predictions.view(B, self.num_anchors, 6, H, W)[:, :, :2, :, :]
    offsets = predictions.view(B, self.num_anchors, 6, H, W)[:, :, 2:, :, :]
    if mode == 'train':
      anchor_idx = torch.cat((pos_anchor_idx, neg_anchor_idx), 0)
      conf_scores = self._extract_anchor_data(conf_scores, anchor_idx)
      offsets = self._extract_anchor_data(offsets, pos_anchor_idx)
      M, _ = offsets.shape
      proposals = GenerateProposal(pos_anchor_coord.view(1, 1, 1, M, 4), offsets.view(1, 1, 1, M, 4), method='FasterRCNN')
      proposals = proposals.view(-1, 4)
    if mode == 'train':
      return conf_scores, offsets, proposals
    elif mode == 'eval':
      return conf_scores, offsets

class FeatureExtractor(nn.Module):
  def __init__(self, reshape_size=224, pooling=False, verbose=False):
    super().__init__()

    from torchvision import models
    from torchsummary import summary

    self.mobilenet = models.mobilenet_v2(pretrained=True)
    self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1]) # Remove the last classifier

    # average pooling
    if pooling:
      self.mobilenet.add_module('LastAvgPool', nn.AvgPool2d(math.ceil(reshape_size/32.))) # input: N x 1280 x 7 x 7

    for i in self.mobilenet.named_parameters():
      i[1].requires_grad = True # fine-tune all

    if verbose:
      summary(self.mobilenet.cuda(), (3, reshape_size, reshape_size))
  
  def forward(self, img, verbose=False):
    num_img = img.shape[0]
    
    img_prepro = img

    feat = []
    process_batch = 500
    for b in range(math.ceil(num_img/process_batch)):
      feat.append(self.mobilenet(img_prepro[b*process_batch:(b+1)*process_batch]
                              ).squeeze(-1).squeeze(-1)) # forward and squeeze
    feat = torch.cat(feat)
    
    if verbose:
      print('Output feature shape: ', feat.shape)
    
    return feat

class RPN(nn.Module):
  def __init__(self):
    super().__init__()

    # READ ONLY
    self.anchor_list = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]])
    self.feat_extractor = FeatureExtractor()
    self.prop_module = ProposalModule(1280, num_anchors=self.anchor_list.shape[0])

  def forward(self, images, bboxes, output_mode='loss'):
    # weights to multiply to each loss term
    w_conf = 1 # for conf_scores
    w_reg = 5 # for offsets

    assert output_mode in ('loss', 'all'), 'invalid output mode!'
    total_loss = None
    conf_scores, proposals, features, GT_class, pos_anchor_idx, anc_per_img = \
      None, None, None, None, None, None
    features = self.feat_extractor(images)
    grid_list = GenerateGrid(images.shape[0])
    anc_list = GenerateAnchor(self.anchor_list, grid_list).to(bboxes.device, bboxes.dtype)
    iou_mat = IoU(anc_list, bboxes)
    activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, \
    activated_anc_coord, negative_anc_coord = ReferenceOnActivatedAnchors(anc_list, bboxes, grid_list, iou_mat)
    conf_scores, offsets, proposals = self.prop_module(features, activated_anc_coord, activated_anc_ind, negative_anc_ind)
    conf_loss = ConfScoreRegression(conf_scores, features.shape[0])
    reg_loss = BboxRegression(offsets, GT_offsets, features.shape[0])
    anc_per_img = torch.prod(torch.tensor(anc_list.shape[1:-1]))
    total_loss = w_conf * conf_loss + w_reg * reg_loss
    pos_anchor_idx = activated_anc_ind
    if output_mode == 'loss':
      return total_loss
    else:
      return total_loss, conf_scores, proposals, features, GT_class, pos_anchor_idx, anc_per_img


  def inference(self, images, thresh=0.5, nms_thresh=0.5, mode='RPN'):
    assert mode in ('RPN', 'FasterRCNN'), 'invalid inference mode!'
    # import pdb;pdb.set_trace()
    features, final_conf_scores, final_proposals = None, None, None
    final_conf_probs, final_proposals = [], []
    features = self.feat_extractor(images)
    grid_list = GenerateGrid(images.shape[0])
    anc_list = GenerateAnchor(self.anchor_list, grid_list).to(images.device, images.dtype)
    conf_scores, offsets = self.prop_module(features)
    offsets = offsets.permute(0, 1, 3, 4, 2)
    conf_scores = torch.sigmoid(conf_scores)
    anchors_proposal = GenerateProposal(anc_list, offsets, method='FasterRCNN')
    B, A, _, H, W = conf_scores.shape
    for b in range(B):
      boxes = []
      scores = []
      for a in range(A):
        for h in range(H):
          for w in range(W):
            if conf_scores[b, a, 0, h, w] >= thresh:
              boxes.append(anchors_proposal[b, a, h, w, :].tolist())
              scores.append(conf_scores[b, a, 0, h, w].item())      
      # import pdb;pdb.set_trace()
      boxes = torch.tensor(boxes).to(images.device)
      scores = torch.tensor(scores).to(images.device)
    #   import pdb;pdb.set_trace()
      keep = torchvision.ops.nms(boxes, scores, nms_thresh)
      # keep = nms(boxes, scores, nms_thresh)
      final_proposals.append(boxes[keep].reshape(-1, 4))
      final_conf_probs.append(scores[keep].reshape(-1, 1))
    if mode == 'RPN':
      features = [torch.zeros_like(i) for i in final_conf_probs] # dummy class
    return final_proposals, final_conf_probs, features

class TwoStageDetector(nn.Module):
  def __init__(self, in_dim=1280, hidden_dim=256, num_classes=20, \
               roi_output_w=2, roi_output_h=2, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0)
    self.num_classes = num_classes
    self.roi_output_w, self.roi_output_h = roi_output_w, roi_output_h
    self.RPN = RPN()
    self.region_classification = nn.Sequential(
        nn.Linear(in_dim, hidden_dim*2),
        nn.Dropout(p=drop_ratio),
        nn.ReLU(),
        nn.Linear(hidden_dim*2, hidden_dim),
        nn.Dropout(p=drop_ratio),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, num_classes)
    )

  def forward(self, images, bboxes):
    B, _, H, W = images.shape
    rpn_loss, conf_scores, proposals, features, GT_class, pos_anchor_idx, anc_per_img = self.RPN(images, bboxes, output_mode='all')
    H_prime = features.shape[2]
    batch_index = (pos_anchor_idx // anc_per_img).view(-1, 1).to(proposals.dtype)
    proposals_index = torch.cat((batch_index, proposals), 1)
    roi_feature = torchvision.ops.roi_align(features, proposals_index, (self.roi_output_h, self.roi_output_w))
    M = GT_class.shape[0]
    mean_pool = torch.nn.AvgPool2d(2)
    roi_feature = mean_pool(roi_feature).view(M, -1)
    class_probs = self.region_classification(roi_feature)
    cls_loss = F.cross_entropy(class_probs, GT_class, reduction='sum') * 1. / M
    total_loss = rpn_loss + cls_loss
    return total_loss

  def inference(self, images, thresh=0.5, nms_thresh=0.7):
    final_proposals, final_conf_probs, final_class = None, None, None
    final_class = []
    B, _, H, W = images.shape
    # print("hello2")
    # import pdb;pdb.set_trace()
    final_proposals, final_conf_probs, features = self.RPN.inference(images, thresh, nms_thresh, mode='FasterRCNN')
    for b in range(B):
      proposal = final_proposals[b]
      if proposal.shape[0] == 0:
        final_class.append(torch.tensor([]).to(images.device).reshape(-1, 1))
        continue
      H_prime = features.shape[2]
      index = torch.tensor([b], dtype=images.dtype, device=images.device).view(-1, 1).expand(proposal.shape[0], -1)
      proposal_index = torch.cat((index, proposal), 1)
      roi_feature = torchvision.ops.roi_align(features, proposal_index, (self.roi_output_h, self.roi_output_w))
      mean_pool = torch.nn.AvgPool2d(self.roi_output_h)
      roi_feature = mean_pool(roi_feature).view(proposal.shape[0], -1)
      class_probs = self.region_classification(roi_feature)
      _, class_index = torch.max(class_probs, 1)
      final_class.append(class_index.reshape(-1, 1))
    return final_proposals, final_conf_probs, final_class

def nms(boxes, scores, iou_threshold=0.5, topk=None):
  if (not boxes.numel()) or (not scores.numel()):
    return torch.zeros(0, dtype=torch.long)

  keep = None
  keep = []
  x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
  areas = (x2 - x1) * (y2 - y1)
  sorted, order = torch.sort(scores, descending=True)
  while order.shape[0] > 0:   
    i = order[0]
    keep.append(i)
    xx1 = torch.max(x1[i], x1[order[1:]])
    yy1 = torch.max(y1[i], y1[order[1:]])
    xx2 = torch.min(x2[i], x2[order[1:]])
    yy2 = torch.min(y2[i], y2[order[1:]])
    intersection = torch.max(torch.zeros(1), xx2 - xx1) * torch.max(torch.zeros(1), yy2 - yy1)
    iou = intersection / (areas[i] + areas[order[1:]] - intersection)
    index = torch.squeeze(torch.nonzero((iou <= iou_threshold)) + 1)
    order = order[index]
    order = order.reshape(-1)
  if topk is not None:
    keep = keep[: topk]
  
  keep = torch.tensor(keep)
  keep.to(boxes.device)
  return keep

def ConfScoreRegression(conf_scores, batch_size):
  # the target conf_scores for positive samples are ones and negative are zeros
  M = conf_scores.shape[0] // 2
  GT_conf_scores = torch.zeros_like(conf_scores)
  GT_conf_scores[:M, 0] = 1.
  GT_conf_scores[M:, 1] = 1.

  conf_score_loss = F.binary_cross_entropy_with_logits(conf_scores, GT_conf_scores, \
                                     reduction='sum') * 1. / batch_size
  return conf_score_loss

def BboxRegression(offsets, GT_offsets, batch_size):
  bbox_reg_loss = F.smooth_l1_loss(offsets, GT_offsets, reduction='sum') * 1. / batch_size
  return bbox_reg_loss