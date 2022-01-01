## Traffic Sign Detection Using Mask RCNN


##### Project Aim

Traffic sign detection is an essential capibility required by self-driving cars in today's age. A model that does not detect traffic signs in all conditions can lead to confusion and eventual road mishaps. 

This project is aimed at enhancing the robustness of the mask rcnn model in detecting traffic signs. 

#### Methodology 

The method to tweak the mask rcnn model to better perform on traffic sign data set has been described in the paper https://arxiv.org/pdf/1904.00649.pdf

This project further  

### Proposed data augmentations to increate model robustness

![augmentatios](https://github.com/adriannaStaszewska/CS539Project/blob/main/assets/augmentations.jpg)

### Detection after training the data on new dataset with transfer learning from COCO weights

![detection](https://github.com/adriannaStaszewska/CS539Project/blob/main/assets/detection.jpg)


### Detection after training the model on augmented data and reducing the polygon size 

![detection2](https://github.com/adriannaStaszewska/CS539Project/blob/main/assets/detection2.jpg)
