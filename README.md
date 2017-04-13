# SSD: Single Shot Multibox Detector (Tensorflow)

Fine-tuned to Caltech Pedestrian Dataset. Achieved L-AMR reasonable close to 13.06, (as found in https://arxiv.org/pdf/1610.03466.pdf)

Extended to hand-labelled vehicle dataset

Improved upon using tracking information provided by video

## DONE:
(caltech_to_tfrecords.py, caltech_common.py) - Can create tfrecord files, but still need to make appropriate config (Person_full and Person_occluded) - This is for legit, so make new files and edit these for conversion to tfrecords for the BS strategy

## TODO (BS first to keep in back pocket, then try best at legit):

### Training:

- Convert Caltech Training Set (Set00 - Set05) to tfrecord files for training (Annotations (JSON) has been extracted for all sets but Frames (JPEG) have only been extracted for Set01). 
  - caltech_to_tfrecords.py functions at creating tfrecords, but configurations need to be done to match with the Fused DNN "state of the art" baseline. 
    - We need to use only images that contain at least one annotated pedestrian (both original and flipped images)
    - Only use 'Person' and 'People'. Further classify 'Person' into 'Person_full' and 'Person_occluded' 
      - Occluded pedestrians were annotated with two BBs that denote  the  visible  and  full  pedestrian  extent.
      - For  each  occluded  pedestrian,  we  can  compute  the fraction of occlusion as one minus the visible pedestrian area            divided  by  total  pedestrian  area  (calculated  from the  visible  and  full  BBs).
      - Over  80%  occlusion  typically indicates full occlusion, while 0% is used to indicate that a BB could not represent the extent of the visible region (invalid)
      - We  further  subdivide the  cases  in  between  into partial occlusion  (1-35%  area occluded) and heavy occlusion (35-80% occluded)
- Setup Default BBs (This is requirement if we use pre-trained SSD on Pascal, as Fused DNN used pre-trained SSD on Coco, and in SSD paper they state that as  objects in COCO tend to be smaller than PASCAL VOC, they use smaller default boxes for all layers)
  - We place 7 default BBs with aspect ratios [0.1, 0.2, 0.41a, 0.41b, 0.8, 1.6, 3.0] on top of each location of all output feature maps
  -  All default BBs except 0.41b have relative heights [0.05, 0.1, 0.24, 0.38, 0.52, 0.66, 0.80] for the 7 output layers. The heights for 0.41b are [0.1,  0.24,  0.38,  0.52,  0.66,  0.80,  0.94]
- Fine-tune SSD Model (Confused about this; In the paper they said they used a SSD pre-trained on COCO, but all the layers after each output layer are randomly initialized and trained from scratch. What this tells me is solely the VGG-16 was pre-trained on COCO. As in the original SSD paper, the same architecture that was applied to COCO was applied to ImageNet, and ImageNet is the best, so we should 100% follow the instructions underneath "Fine-tuning a network trained on ImageNet" in the original README)
  - train_ssd_network.py
  - 40K iterations using SGD, Learning rate 10-5

* BS Possibility: Rather than doing this properly like the above, simply straight up use the Pascal VOC classes, ignore all classes outside of 'Person', maybe only classify Person_full as Person for guaranteeing good results or also classify Person_occluded in the PASCAL way by just marking it as truncated, and straight up use a pre-trained VGG-300 SSD network, like it says under "Fine-tuning existing SSD checkpoints". Let's do this if all else fails with actually doing this properly like the above. 

### Testing:
- Compute the log-average miss rate (L-AMR) as the performance metric
  -  L-AMR is computed evenly spaced in log-space in the range 10−2 to 10-0 by averaging miss rate at the rate of nine false positives per image (FPPI)
  - For curves that end before reaching a given FPPI rate, the minimum miss rate achieved is used
  - Similar to the PASCAL average precision as it represents the entire  curve by a single reference value.
  - eval_ssd_network.py provides estimates on the recall-precision curve and compute the mAP metrics following the Pascal VOC 2007 and 2012 guidelines. So this will need to be changed to also give L-AMR
- Evaluate on the 'Reasonable' Evaluation Setting
  - Height: Over 50 pixels
  - Occlusion: No or Partial occlusion
  - Filter only these images out of Caltech Test Set (Set06 - Set10)

* BS Possibility: Just evaluate using the script, and get out MAP and precision-recall. Forget the baseline...

### Vehicle:
- Split into training and test
- Convert train to tfrecord, Setup default BBs (bigger), Fine-tune SSD model
- Figure out evaluation setting, if any
- Set baseline with our vehicle dataset

* BS Possibility: Just do same thing as for caltech, except now ignore all classes outside of 'car', 'bus', and 'motorbike'

### Improvement:
- Use the fact that we have videos (for both pedestrians and vehicles) to improve (Ihsan Probability Model) (TEMPORAL INTEGRATION)
- As suggested in SSD paper, "explore its use as part of a system using recurrent neural networks to detect and track objects in video simultaneously" 
- Maybe look into the Fused DNN and the Caltech Pedestrian Paper suggestions

* BS Possibility: Try something, and if it doesn't work suggest further work
