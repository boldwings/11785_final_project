# Introduction
In our project, we aim to accelerate Faster R-CNN object detection task to further exploit deep neural networks' potential in real-time object detection. We attempted different backbone structures and modified part of the Region Proposal Network (RPN) based on the original implementation of Faster R-CNN. We tried to replace the original VGG-16 backbone of the network with lighter-weight network such as MobileNet, and compensate the loss in detection accuracy with our implementation of Region of Interest (RoI) Alignment. 

# Dataset
Our model is trained with PASCAL VOC2007 dataset. This dataset is officially released for the PASCAL Visual Object Classes Challenge 2007. It contains 9,963 images of different sizes. All the images are manually annotated with boxes to indicate object locations. An example image with annotated boxes. Provided images also vary in complexity, objects may be overlapped and truncated. There are 24,640 annotated objects from 20 classes in total. All classes belongs to one of Person, Animal, Vehicle and Indoor categories. 

We choose to use this dataset because it is widely used to train and validate object detection architectures. In this way, we can compare our model with other architectures. Due to time limits, the amount of data contained in this dataset fits our timeline. We can also add data points from PASCAL VOC2012 to expand out dataset and reduce class-imbalance problems.

# Train the Model
We provide 2 options to run our code.
## Using Google Colab
Since the code development is mostly done on Google Colab, it's recommeded to check our implementation in Accelearting_faster_rcnn.ipynb and run each code cell in sequence to prepare the data and train the model. The last 3 cells are for training, testing and accuracy calculation respectively.
To train the model, run the first of the last 3 cells
```
train_loader = pascal_voc2007_loader(train_dataset, 100) # a new loader
num_epochs = 50
lr = 5e-3
frcnn_detector = TwoStageDetector()
DetectionSolver(frcnn_detector, train_loader, learning_rate=lr, num_epochs=num_epochs)
model_save_name = 'frcnn_detector.pt'
path = F"/content/gdrive/My Drive/{model_save_name}" 
torch.save(frcnn_detector.state_dict(), path)
```
To do the inference, run the following cell:
```
frcnn_detector = TwoStageDetector()
model_save_name = 'frcnn_detector.pt'
path = F"/content/gdrive/My Drive/{model_save_name}"
frcnn_detector.load_state_dict(torch.load(path))
DetectionInference(frcnn_detector, small_train_loader, small_dataset, idx_to_class, thresh=0.9)
```
Finally, to test the accuracy, run the following cell
```
!rm -r mAP/input/*
DetectionInference(frcnn_detector, train_loader, train_dataset, idx_to_class, output_dir='mAP/input', thresh=0.8, nms_thresh=0.3) 
!cd mAP && python main.py
```
## Use python files
If you are more comfortable with pure python, you can also clone our repo and follow the directions below to train your model:
1. run ```wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar``` to get the train and validation dataset
2. run ```wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar``` to get the test dataset
3. run ```tar -xvf filename``` to untar 2 files
4. run ```python3 train.py``` to train the model

# Parameter Tuning
1. To change the size of anchors, modify ```anchor_list``` in ```train.py```. Currently we are using 9 anchors with size [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3].

2. To adjust the threshold for positive anchors, change ```thresh=0.7``` in the ```DetectionInference``` function

3. To adjust nms threshold for overlapping bounding boxes, change ```nms_thresh=0.3``` in the ```DetectionInference``` function

4. To use a pretrained backbone model, navigate to the ```FeatureExtractor``` function in ```models.py``` and change the backbone model accordingly.
