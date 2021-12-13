# CS539 Group Project

## TODO

## Dataset

## Data Pruning

The dataset was very large, and this greatly slowed Google Collab's ability to train a model on our data.  To remedy this, we pruned the data several different ways.

The first way we pruned the data was removing extraneous categories.  Many of the categories of the TT100K dataset have very small amounts of instances, or none at all.  To this end, we removed these categories by removing their annotations within the data.  While experimenting, we also explored the effect of removing images with these problematic categories instead.

We then split the dataset based on the divisions made in the dataset (train/other/test) and sought to cut down categories with excessively disproportionate intances.  To do this, sorted the images based on number of offending categories, then greedily removed them.  While greedily removing images, the algorithm passed over images that had rarer categories to ensure there remained enough instances.  Because of the combinatorial nature of this problem, a greedy search was used to avoid any np algorithms.

Finally, we removed a percentage of the images that were in the background category (images with no objects whatsoever).  This removed many images, especially from the other segment of the dataset.

## Data Augmentation

### Purpose
Our approach focused on augmenting the dataset, to make the resulting model better equipped to deal with real driving situations.
TODO

### Process (TBD, I'm speculating on this)
Before we utilized any of the data augmentation, we first trained a model on the base dataset.  We started with those weights when running further training using data augmentation.

For each epoch thereafter, we took a bootstrap sample of the dataset, then selected several of the images to be transformed with our augmentation effects (outlined below).  The epoch would then train on this data.

### Motion Blur
The first technique we used to augment the data was motion blur.  This was done because while a vehicle is in motion, a lower quality camera may take a blurry image.  To simulate this, we created a motion blur effect using cv2 kernels.  The strength of the effect actually scales with the speed limit of the image.  As location metadata was not provided within the dataset, the presence of speed signs were used instead.  The highest present speed sign within an image is what the 'speed' of the image is.

If no such speed signs exist, then a speed is sampled from a normal distribution. This normal distribution has mean and standard deviation derived from the subset of images containing speed signs within the data.

### Rain Particles
TODO
### TODO
TODO

## Results

TODO

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
