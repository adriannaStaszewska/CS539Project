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

![augmentatios](https://github.com/deveshdatwani/traffic-sign-detection-using-mask-rcnn/blob/main/assets/augmentations.jpg)

##### Detection after training the data on new dataset with transfer learning from COCO weights

![detection](https://github.com/deveshdatwani/traffic-sign-detection-using-mask-rcnn/blob/main/assets/detection.jpg)


##### Detection after training the model on augmented data and reducing the polygon size 

![detection2](https://github.com/deveshdatwani/traffic-sign-detection-using-mask-rcnn/blob/main/assets/detection2.jpg)


##### Data pruning

For this project, we aimed to train the mask-rcnn model on a data set provided by Tsinghua Universit, China. Reader can learn more about the data set here  https://cg.cs.tsinghua.edu.cn/traffic-sign/


Consequently, due the shear size of the dataset, we carried out some pruning on the following conditions;

1. remove categories with small amount of instances
2. cut down categories with excessively disproportionate instances.
3. remove background noise


![datapruning](https://github.com/deveshdatwani/traffic-sign-detection-using-mask-rcnn/blob/main/assets/data_pruning.png)

##### Model tweaks

Additionaly, we tweaked our model as per the following paper https://arxiv.org/pdf/1904.00649.pdf, for the Mask-RCNN model to adapt for traffic sign detection. The following methods are proposed;

1. Online hard-example mining
2. Distribution of selected training samples
3. Sample weighting
4. Adjusting region pass-through during detection


##### Training

![training](https://github.com/deveshdatwani/traffic-sign-detection-using-mask-rcnn/blob/main/assets/training.png)

We can see the loss values decreasing with more epochs, but we had to limit the number of epochs due to limite resources on Google Colab. 


#### Results 

##### Low mAP at first

The initial results on test dataset didn't look promising with high losses and an mAP value of **0.045**. 

##### The problem! 

![low_map](https://github.com/deveshdatwani/traffic-sign-detection-using-mask-rcnn/blob/main/assets/low_map.png)

The problem lied with the ellipse/polygon sizes defined in the dataset. These were not tightly fit to the instances. Hence, it was our belief that the model was learning the background as well. 

We tweaked the dataset for the ellipses to now fit tightly to the instances. 


##### mAP values rose by 5.5 times! 

While the mAP values rose by more then 5 times, a lot more work is required to detect traffic signs better. 

![high_map](https://github.com/deveshdatwani/traffic-sign-detection-using-mask-rcnn/blob/main/assets/high_map.png)



