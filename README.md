# RealTime Hyperface
A neural network architecture for realtime simultaneous face detection, landmark localization, pose estimation and gender recognition. 

This work was inspired by the following two publicationss:
* [HyperFace: A Deep Multi-task Learning Framework for Face Detection, Landmark Localization, Pose Estimation, and Gender Recognition](https://arxiv.org/pdf/1603.01249.pdf) 
* [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)


## DEMO
* [Roll](https://drive.google.com/open?id=0B_bGPmnvECTmWlBaT1EtTUN4d2c)
* [Pitch](https://drive.google.com/open?id=0B_bGPmnvECTmaHk5c3R6Z3NNc00)
* [Yaw](https://drive.google.com/open?id=0B_bGPmnvECTmem81MkJfMy0zS1U)

## DATASET
[Annotated Facial Landmarks in the Wild (AFLW)](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)


## Network Architecture
<img src="https://github.com/Adityav2410/RealTimeHyperface/blob/master/assets/images/architecture.png" width=500 align="middle" >


### RPN Vizualization 
<img src="https://github.com/Adityav2410/RealTimeHyperface/blob/master/assets/images/rpn.png" width=500 align="middle" >


### Layer vizualization
<img src="https://github.com/Adityav2410/RealTimeHyperface/blob/master/assets/images/roi_face.png" width=500 align="middle" >

 +
 +
<img src="https://github.com/Adityav2410/RealTimeHyperface/blob/master/assets/images/roi_group.png" width=500 align="middle" >




## RESULTS

<img src="https://github.com/Adityav2410/RealTimeHyperface/blob/master/assets/images/annotated_images.png" width=500 align="middle" >

<img src="https://github.com/Adityav2410/RealTimeHyperface/blob/master/assets/images/regr_improve.png" width=500 align="middle" >

### Face Detection
<img src="https://github.com/Adityav2410/RealTimeHyperface/blob/master/assets/images/detection.png" width=500 align="middle" >

### Gender Recognition
<img src="https://github.com/Adityav2410/RealTimeHyperface/blob/master/assets/images/gender.png" width=500 align="middle" >

### Pose Estimation
<img src="https://github.com/Adityav2410/RealTimeHyperface/blob/master/assets/images/roll.png" width=500 align="middle" >
<img src="https://github.com/Adityav2410/RealTimeHyperface/blob/master/assets/images/pitch.png" width=500 align="middle" >
<img src="https://github.com/Adityav2410/RealTimeHyperface/blob/master/assets/images/yaw.png" width=500 align="middle" >

We have achieved a frame rate of 4FPS on Nvidia Geforce GTX TITAN X GPU.

## REPORT
[Project Report](https://drive.google.com/a/eng.ucsd.edu/file/d/0B8fEfjvUe2O-TnBObUpaOFJxejA/view?usp=sharing)
