# RealTime Hyperface
A neural network architecture for realtime simultaneous face detection, landmark localization, pose estimation and gender recognition. 

This work was inspired by the following two works:
* [HyperFace: A Deep Multi-task Learning Framework for Face Detection, Landmark Localization, Pose Estimation, and Gender Recognition](https://arxiv.org/pdf/1603.01249.pdf") 
* [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)

## PROBLEM STATEMENT DESCRIPTION
Human activities are monitored with the help of Smartphone sensors(Acclerometer and Gyroscope). The statement is to classify the human activities into one of 12 classes based on these sensor readings. 

<img src="https://github.com/Adityav2410/RealTimeHyperface/blob/master/assets/images/architecture.png" width=500 align="middle" >


## DATASET
[Annotated Facial Landmarks in the Wild (AFLW)]("https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/")

The smartphone sensor data are transformed into two categories:- 
*  Time Domain Features - Acclearation(x,y,x), min, median, entropy, etc. 

*  Frequency Domain Features - DFT of time domain features(accleration, jerk magnitude, gyroscope magnitude, etc).


### Visualization 
Data is visualized using 2-D PCA and TSNE embeddings. TSNE visualization shows that the different classes are well seperable. 

<img src="https://github.com/Adityav2410/RealTimeHyperface/blob/master/assets/images/RPN_Heatmap1.png" width=500 align="middle" >

<img src="https://github.com/Adityav2410/RealTimeHyperface/blob/master/assets/images/RPN_Heatmap2.png" width=500 align="middle" >



## RESULTS

<img src="https://github.com/Adityav2410/RealTimeHyperface/blob/master/assets/images/correctClassification.png" width=500 align="middle" >

<img src="https://github.com/Adityav2410/RealTimeHyperface/blob/master/assets/images/failureScenario.png" width=500 align="middle" >


Several classification techniques are implemented across different parameter variation. A detailed study of all the experiment as mentioned below are presented: 

* Neural Network(Single and Multilayer perceptron)
* SVM(Linear and Gaussian Kernel)
* Boosting(with different loss functions)

### Single Layer Neural Network

| Training Accuracy(%)| Validation Accuracy(%) | Test Accuracy(%) | 
|:-------------------:|:----------------------:| ----------------:|
|        97.55        |        96.2            |       92.13      |



### Multilayer Neural Network

| Number of hidden units|Training Accuracy(%) | Validation Accuracy(%) | Test Accuracy(%) | 
|:---------------------:|:-------------------:|:----------------------:|:----------------:|
|          128          |        98.28        |        97.17           |       93.17      |
|          256          |        99.03        |        97.04           |       93.39      |
|          512          |        99.51        |        97.94           |       93.48      |



### L2- SVM


| Kernel   |        Parameters     | Training Accuracy(%)| Validation Accuracy(%) | Test Accuracy(%) | 
| ---------|:---------------------:|:-------------------:|:----------------------:|:----------------:|
| Linear   |          C = 1        |        99.53        |        96.98           |       95.19      |
| Gaussian | C = 5000, gamma = 1e-5|        98.83        |        96.6            |       94.4       |


<img src="https://github.com/Adityav2410/HAPT-Recognition/blob/master/assets/images/pc_Acc_SVM.png" width=500 align="middle" >


### Boosting

| Loss Function |  Weak learners  | Number of weak learner | Training Accuracy(%)| Validation Accuracy(%)|Test Accuracy(%)| 
| ------------- | --------------- |:----------------------:|:-------------------:|:---------------------:|:--------------:|
| Exponential   | Decision Stumps |          339           |        99.97        |        95.6           |       91.68         |   
| Cross Entropy | Decision Stumps |          303           |        99.41        |        94.21          |       91.4          |


