## Traffic Sign Detection Using Mask RCNN


#### Project Aim

Traffic sign detection is an essential capibility required by self-driving cars in today's age. A model that does not detect traffic signs in all conditions can lead road mishaps. 

This project is aimed at enhancing the robustness of the mask rcnn model in detecting traffic signs. 

#### Methodology 

The method to tweak the mask rcnn model to better perform on traffic sign data set has been described in the paper https://arxiv.org/pdf/1904.00649.pdf

This project builds on the above paper.

We introduce data augmentations to support camera failures in the events of motion blur, condensation and random noise.

These data augmentations are implemented with the OpenCV library.   

##### Proposed data augmentations to increate model robustness

![augmentatios](https://github.com/adriannaStaszewska/CS539Project/blob/main/assets/augmentations.jpg)

##### Detection after training the data on new dataset with transfer learning from COCO weights

![detection](https://github.com/adriannaStaszewska/CS539Project/blob/main/assets/detection.jpg)


##### Detection after training the model on augmented data and reducing the polygon size 

![detection2](https://github.com/adriannaStaszewska/CS539Project/blob/main/assets/detection2.jpg)


##### Data pruning

For this project, we aimed to train the mask-rcnn model on a data set provided by Tsinghua Universit, China. Reader can learn more about the data set here  https://cg.cs.tsinghua.edu.cn/traffic-sign/


Consequently, due the shear size of the dataset, we carried out some pruning on the following conditions;

1. remove categories with small amount of instances
2. cut down categories with excessively disproportionate instances.
3. remove background noise


![datapruning]

#####