from dataloader import *
from util import *
from models import *

def plot_loss():
    loss_history = [15.54, 11.42, 9.58, 8.40, 7.49, 6.78, 6.38, 5.85, 5.57, 5.25,
                    5.14, 4.93, 4.69, 4.47, 4.39, 4.25, 4.21, 4.03, 3.93, 3.80,
                    3.67, 3.64, 3.63, 3.56, 3.42, 3.30, 3.24, 3.22, 3.17, 3.12,
                    3.05, 2.97, 2.89, 2.83, 2.80, ]
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.savefig('train_loss.png')

def DetectionSolver(detector, train_loader, learning_rate=3e-3,
                    lr_decay=1, num_epochs=20, **kwargs):

  # ship model to GPU
  detector.to(**to_float_cuda)

  
  # optimizer = optim.Adam(
  optimizer = optim.SGD(
    filter(lambda p: p.requires_grad, detector.parameters()),
    learning_rate) # leave betas and eps by default
  lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                             lambda epoch: lr_decay ** epoch)

  # sample minibatch data
  loss_history = []
  detector.train()
  for i in range(num_epochs):
    start_t = time.time()
    cur_loss = 0
    for iter_num, data_batch in enumerate(train_loader):
      images, boxes, w_batch, h_batch, _ = data_batch
      resized_boxes = coord_trans(boxes, w_batch, h_batch, mode='p2a')
      images = images.to(**to_float_cuda)
      resized_boxes = resized_boxes.to(**to_float_cuda)

      loss = detector(images, resized_boxes)
      optimizer.zero_grad()
      loss.backward()
      cur_loss += loss.item()
      
      optimizer.step()

      # print('(Iter {} / {})'.format(iter_num, len(train_loader)))

    end_t = time.time()
    loss_history.append(cur_loss / len(train_loader))
    print('(Epoch {} / {}) loss: {:.4f} time per epoch: {:.1f}s'.format(
        i, num_epochs, loss_history[-1], end_t-start_t))

    lr_scheduler.step()

  # plot the training losses
  plt.plot(loss_history)
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.title('Training loss history')
  plt.savefig('train_loss.png')

def DetectionInference(detector, data_loader, dataset, idx_to_class, thresh=0.8, nms_thresh=0.3, output_dir=None):

  # ship model to GPU
  detector.to(**to_float_cuda)
 
  detector.eval()
  start_t = time.time()

  if output_dir is not None:
    det_dir = 'mAP/input/detection-results'
    gt_dir = 'mAP/input/ground-truth'
    if os.path.exists(det_dir):
      shutil.rmtree(det_dir)
    os.mkdir(det_dir)
    if os.path.exists(gt_dir):
      shutil.rmtree(gt_dir)
    os.mkdir(gt_dir)

  for iter_num, data_batch in enumerate(data_loader):
    # debug: print something
    # print(data_batch.shape)

    images, boxes, w_batch, h_batch, img_ids = data_batch
    images = images.to(**to_float_cuda)
    # print("hello")
    final_proposals, final_conf_scores, final_class = detector.inference(images, thresh=thresh, nms_thresh=nms_thresh)

    # clamp on the proposal coordinates
    batch_size = len(images)
    for idx in range(batch_size):
      torch.clamp_(final_proposals[idx][:, 0::2], min=0, max=w_batch[idx])
      torch.clamp_(final_proposals[idx][:, 1::2], min=0, max=h_batch[idx])

      # visualization
      # get the original image
      # hack to get the original image so we don't have to load from local again...
      i = batch_size*iter_num + idx
      img, _ = dataset.__getitem__(i)

      valid_box = sum([1 if j != -1 else 0 for j in boxes[idx][:, 0]])
      final_all = torch.cat((final_proposals[idx], \
        final_class[idx].float(), final_conf_scores[idx]), dim=-1).cpu()
      resized_proposals = coord_trans(final_all, w_batch[idx], h_batch[idx])

      # write results to file for evaluation (use mAP API https://github.com/Cartucho/mAP for now...)
      if output_dir is not None:
        file_name = img_ids[idx].replace('.jpg', '.txt')
        with open(os.path.join(det_dir, file_name), 'w') as f_det, \
          open(os.path.join(gt_dir, file_name), 'w') as f_gt:
          print('{}: {} GT bboxes and {} proposals'.format(img_ids[idx], valid_box, resized_proposals.shape[0]))
          for b in boxes[idx][:valid_box]:
            f_gt.write('{} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(idx_to_class[b[4].item()], b[0], b[1], b[2], b[3]))
          for b in resized_proposals:
            f_det.write('{} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(idx_to_class[b[4].item()], b[5], b[0], b[1], b[2], b[3]))
        # import pdb;pdb.set_trace()
        imgcopy = data_visualizer(img, idx_to_class, boxes[idx][:valid_box], resized_proposals)
        plt.imsave("./result_images/" + img_ids[idx], imgcopy)
  end_t = time.time()
  print('Total inference time: {:.1f}s'.format(end_t-start_t))

def overfit_small_data():
    for lr in [1e-3]:
        print('lr: ', lr)
        rpn = RPN()
        RPNSolver(rpn, small_train_loader, learning_rate=lr, num_epochs=400)

    torch.save(rpn.state_dict(), 'rpn_small')  

def rpn_inference(rpn):
    RPNInference = DetectionInference
    RPNInference(rpn, small_train_loader, small_dataset, idx_to_class, thresh=0.8, nms_thresh=0.3)

def overfit_rcnn():
    lr = 1e-3
    detector = TwoStageDetector()
    DetectionSolver(detector, small_train_loader, learning_rate=lr, num_epochs=400)
    torch.save(detector.state_dict(), 'detector_small')


# data type and device for torch.tensor
to_float = {'dtype': torch.float, 'device': 'cpu'}
to_float_cuda = {'dtype': torch.float, 'device': 'cuda'}
to_double = {'dtype': torch.double, 'device': 'cpu'}
to_double_cuda = {'dtype': torch.double, 'device': 'cuda'}
to_long = {'dtype': torch.long, 'device': 'cpu'}
to_long_cuda = {'dtype': torch.long, 'device': 'cuda'}

train_dataset = get_pascal_voc2007_data('./', 'train')
val_dataset = get_pascal_voc2007_data('./', 'val')
test_dataset = get_pascal_voc2007_data('./', 'test')
class_to_idx = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,
                'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9, 'diningtable':10,
                'dog':11, 'horse':12, 'motorbike':13, 'person':14, 'pottedplant':15,
                'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19
}
# import pdb;pdb.set_trace()
idx_to_class = {i:c for c, i in class_to_idx.items()}

train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(0, 2500)) # use 2500 samples for training
train_loader = pascal_voc2007_loader(train_dataset, 15, num_workers=4, isTrain=True)
val_loader = pascal_voc2007_loader(val_dataset, 10, num_workers=4)
test_loader = pascal_voc2007_loader(test_dataset, 50, num_workers=4)
train_loader_iter = iter(train_loader)
img, ann, _, _, _ = train_loader_iter.next()

anchor_list = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]], **to_float_cuda)




RPNSolver = DetectionSolver 
num_sample = 40
small_dataset = torch.utils.data.Subset(train_dataset, torch.linspace(0, len(train_dataset)-1, steps=num_sample).long())
small_train_loader = pascal_voc2007_loader(small_dataset, 200) # a new loader

# overfit_rcnn()
# overfit_small_data()
# pretrained_dict_rpn = torch.load('rpn_small')
# rpn = RPN()
# rpn.load_state_dict(pretrained_dict_rpn)

# pretrained_dict_detector = torch.load('detector')
# detector = TwoStageDetector()
# detector.load_state_dict(pretrained_dict_detector)

# train_loader = pascal_voc2007_loader(train_dataset, 100)
# num_epochs = 50
# import pdb;pdb.set_trace()
lr = 1e-3
# pretrained_dict_detector = torch.load('frcnn_complex_10')
frcnn_detector = TwoStageDetector()
# frcnn_detector.load_state_dict(pretrained_dict_detector)
DetectionSolver(frcnn_detector, train_loader, learning_rate=lr, num_epochs=30)
torch.save(frcnn_detector.state_dict(), 'frcnn_complex_15')
# import pdb;pdb.set_trace()

# pretrained_dict_detector = torch.load('frcnn_complex_10')
# frcnn_detector = TwoStageDetector()
# frcnn_detector.load_state_dict(pretrained_dict_detector)
# DetectionSolver(frcnn_detector, train_loader, learning_rate=1e-3, num_epochs=15)
# torch.save(frcnn_detector.state_dict(), 'frcnn_15')
DetectionInference(frcnn_detector, test_loader, test_dataset, idx_to_class, output_dir='mAP/input', thresh=0.7, nms_thresh=0.3)
import pdb;pdb.set_trace()