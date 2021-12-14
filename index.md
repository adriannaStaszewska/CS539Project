# CS539 Group Project

## Abstract
TBD

## Introduction
The locating and identification of traffic signs is key for driving agents to be successful in navigating roadways.  These signs come in varying shapes, sizes, colors, and patterns.  For this project, we explored the usage of the machine learning technique Mask R-CNN on a traffic sign dataset to measure the difference in performance while utilizing custom data augmentation methods.

## Related Work
We gave our paper presentation related to this paper on (Deep Learning for Large-Scale Traffic-Sign
Detection and Recognition)[https://doi.org/10.1109/TITS.2019.2913588].  The dataset within this paper contains a diverse selection of categories.  The paper discusses experiments on an augmented Mask R-CNN.  It also employs data augmentation to expand the size of its data.

## Tools
- [Vizhub](https://vizhub.com/): A website based around creating interactive data visualizations, created by @ckelleher@wpi.edu.  Using this tool, we were able to visualize properties of the dataset.
- [Jupyter Notebook](https://jupyter.org/): A project based around making reproducable and interactive python code with user friendly explanations.
- [Google Colab](https://research.google.com/colaboratory/): We used Google's cloud computing to run our Jupyter Notebook code and train/test our model.
- [Github](https://github.com/): We used github for source code managment and sharing.  We created several different branches using its version control to work on diverging aspects of the project.

## Dataset
Originally, we were going to use the (Mapillary Dataset)[https://www.mapillary.com/dataset/trafficsign] to run our model on, however, this dataset did not come with necessary features.  Namely, it was missing the masks for Mask-RCNN.

The dataset used is instead the TT100K ([Tsinghua-Tencent-100K](https://cg.cs.tsinghua.edu.cn/traffic-sign/)).  This dataset is split into three folders: test, train, and other.  In the parent folder, there is an annotations.json file which contains labels, masks, and bounding boxes for the images in the set.

The TTK100 has many categories of traffic signs (and a background category for images with no objects at all).  There are approximately 10K images in the set with traffic signs in them.  

## Data Pruning
The dataset was very large, and this greatly slowed Google Collab's ability to train a model on our data.  To remedy this, we pruned the data several different ways.

The first way we pruned the data was removing extraneous categories.  Many of the categories of the TT100K dataset have very small amounts of instances, or none at all.  To this end, we removed these categories by removing their annotations within the data.  While experimenting, we also explored the effect of removing images with these problematic categories instead.

We then split the dataset based on the divisions made in the dataset (train/other/test) and sought to cut down categories with excessively disproportionate intances.  To do this, sorted the images based on number of offending categories, then greedily removed them.  While greedily removing images, the algorithm passed over images that had rarer categories to ensure there remained enough instances.  Because of the combinatorial nature of this problem, a greedy search was used to avoid any np algorithms.

Finally, we removed a percentage of the images that were in the background category (images with no objects whatsoever).  This removed many images, especially from the other segment of the dataset.

## Data Augmentation
### Purpose
Our approach focused on augmenting the dataset, to make the resulting model better equipped to deal with real driving situations.  We experimented with different custom-made data augmentation technique subsets to see whether the base Mask-RCNN faired better or worse.

### Process (TBD, I'm speculating on this)
Before we utilized any of the data augmentation, we first trained a model on the base dataset.  We started with those weights when running further training using data augmentation.

For each epoch thereafter, we took a bootstrap sample of the dataset, then selected several of the images to be transformed with our augmentation effects (outlined below).  The epoch would then train on this data.

### Motion Blur
The first technique we used to augment the data was motion blur.  This was done because while a vehicle is in motion, a lower quality camera may take a blurry image.  To simulate this, we created a motion blur effect using cv2 kernels.  The strength of the effect actually scales with the speed limit of the image.  As location metadata was not provided within the dataset, the presence of speed signs were used instead.  The highest present speed sign within an image is what the 'speed' of the image is.

If no such speed signs exist, then a speed is sampled from a normal distribution. This normal distribution has mean and standard deviation derived from the subset of images containing speed signs within the data.

### Rain Particles
TBD

### TBD
TBD

## Model
TBD (Put layers outputs here with captions)

## Methodology
### Mask R-CNN
We trained a Mask R-CNN model within this project.  A brief summary of this technique is outlined within this section.

There are two primary modules within Mask R-CNN:
1. Region Proposal Network: Produces a set of rectangular shaped proposals for regions which may contain a class of interest.
2. Region-Based CNN: Accepts an input region proposal and outputs the class contained within that region.

The model is trained in four steps:
1. The Region Proposal Network is trained using the bounding boxes in the training set.
2. A detector network then gets trained based on the proposals generated by the Region Proposal Network.
3. The R-CNN module is initialized via the detector network and combine the convolution layers.
4. Finally, the fully connected layers of the R-CNN are fine-tuned.

A Feature Pyramid Network (FPN) is also utilized.  This approach is used to improve the model's performance on smaller objects within the image.  To do this, it extracts features from lower layers before downsampling occurs, preventing the loss of significant detail of these objects.

### Training process
TBD

## Results
TBD

# Evaluation Metrics
TBD

## Conclusion
TBD

## Links
- [Code](https://github.com/adriannaStaszewska/CS539Project)
- [Presentation](https://wpi0-my.sharepoint.com/:p:/g/personal/ymao4_wpi_edu/EWkDtaueJTpHhs8x-Wj8HUwBRH5k6N54RdvddHyJOPJRxg?e=GN4w5x)

## Works Cited
TBD

## Team
### Davesh Datwani (dbdatwani@wpi.edu)


### Xingtong Guo (xguo3@wpi.edu)


### Yujun Mao (ymao4@wpi.edu)


### Daniel Ribaudo (dcribaudo@wpi.edu)
I am a Computer Science graduate student with undergraduate degrees in both Computer Science and Interactive Media Game Development from Worcester Polytechnic Institute.  I enjoy fencing, jogging, drinking coffee, and playing video games.

For this project, I worked on data visualization, the website, preprocessing the JSON files / otherwise preparing the data, and motion blur implementation.

### Adrianna Staszewska (azstaszewska@wpi.edu)


# Auto generated Stub

## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/adriannaStaszewska/CS539Project/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/adriannaStaszewska/CS539Project/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
