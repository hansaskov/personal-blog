---
title: 'Two approaches for Kick Detection'
description: 'A guide to detect kick made in a football match'
pubDate: 'Sep 14 2023'
heroImage: '/with_ball_1.png'
---

## Description
This guide covers my attempts how to detect kicks in a football match, either by training a classification model or by using pretrained models to our advantage.

## Objective
The objective is to develop a solution capable of determining whether a kick is taking place in a given frame, with the expectation of a proof-of-concept solution.

## Kick dataset
Before we begin, let's take a closer look at the dataset's structure. The dataset comprises of a football match captured at a rate of one frame per second. The match is split into two folders: `with_ball` and `without_ball`, both of which contain YOLOv5 annotations for bounding boxes around the ball. I initially mistakenly assumed that these bounding boxes represented the players' legs when they kicked the ball, which led to inefficient use of my time. This serves as a reminder to always double-check and ensure a clear understanding of the dataset's contents.

Now, let's proceed to examine some examples from the dataset. Below, you'll find two images, one from the `with_ball` folder and the other from the `without_ball` folder:
#### 1. Image with ball
![example of photo with ball](/with_ball_2.png)

#### 2. Image without ball
![example of photo without ball](/without_ball_1.jpg)




As expected, the first photo contains a ball, while the second one does not. Excellent! Now, before we proceed, it's important to pause for a moment. Upon further inspection of the `without_ball` folder, I encountered some questionable images. Here are two examples that caught my attention:

#### 4. Example of images *"without"* ball
<div style="display: flex; justify-content: space-between;">
    <img src="/without_ball_3.jpg" alt="Kick Image 1" style="width: 47.5%;" />
    <img src="/without_ball_2.jpg" alt="Kick Image 2" style="width: 47.5%;" />
</div>

The first image is incorrectly labeled as it contains a ball, and the second image seems like an error. These examples do not mean the dataset is worthless and should be disregarded; it just means it has not been manually combed through to remove incorrect labels and ensure high quality.

With all of this in mind, let's explore what we've learned from the dataset. First, it's essential to emphasize the importance of double-checking and fully understanding your dataset. Second, not all datasets are equal in quality, and improving dataset quality can significantly boost performance. Lastly, we're still uncertain about how this dataset connects to our set objectives.

## First idea

The main issue with the dataset is that our objective is to detect when a kick has occurred, but no annotations have been provided to train a machine learning model. So what is the only logical solution? You guessed it! Annotate the data yourself.

To save time and keep things simple, I've decided to focus on the 'with_ball' folder and manually tag the images where players are kicking the ball. In that folder, there are 1550 images, but only 64 of them actually show players kicking the ball. I think this number would be much higher if we had more images, as many of the pictures are of players right before or after they kick the ball, not during the kick itself. Below, I've included two examples of images where I believe you can clearly see a kick happening.


#### 5. Examples of kicking the ball

<div style="display: flex; justify-content: space-between;">
    <img src="/kick_2.jpg" alt="Kick Image 2" style="width: 47.5%;" />
    <img src="/kick_1.jpg" alt="Kick Image 1" style="width: 47.5%;" />
</div>

Now that we have curated a dataset, we can start our machine learning approach! 

### Data Splitting

In the realm of machine learning, dividing your dataset into three sets—training, validation, and testing—is pivotal. Here is the diffrence between them

1. Training Set: This is where your model learns from the data, forming the foundation of its knowledge.

2. Validation Set: It's for fine-tuning and optimizing your model. It helps you detect overfitting and refine your model's performance.

3. Test Set: The ultimate reality check, reserved for unbiased evaluation. It ensures your model can make accurate predictions on new, unseen data.

These splits are usually weighted to  70% for training, 20% for validation, and 10% for testing. By adhering to this split, you maintain data integrity, safeguard against overfitting, and ensure your model's real-world readiness. It's a simple yet powerful practice in the journey of machine learning success.

- **Train**
  - Kick
  - No_kick
- **Val**
  - . . .
- **Test**
  - . . .



### Normalize data
Normalization is a vital preprocessing step in data science and machine learning. It's all about getting our data on an even footing before our models start learning. Here's why it's so essential in a nutshell:

1. Equal Treatment: Normalization ensures that all features are treated equally. No feature dominates just because it has a larger scale, leading to a more balanced representation.

2. Efficiency: It helps optimization algorithms, like gradient descent, work better by putting data on a similar scale. This means faster convergence and fewer training iterations.

3. Performance: Normalization reduces the risk of numerical instability and exploding gradients, leading to more stable and accurate models.

In essence, normalization sets the stage for successful machine learning by making sure our data isn't biased by varying scales. So, when working on your next data science project, remember: normalize for better results.

So how do we normalize our dataset? For the rest of the blog i will use pytorch and pytorch lightning as my preferred framework for working with deep learning. But let's get back to it, we will normalize our dataset by going over each of the three RGB channels and calculate the mean and standard deviation for all pixels in our training dataset. 

``` py 
from torchvision import datasets, transforms
from torch import zeros
from torch.utils.data import DataLoader
from tqdm import tqdm

def calc_mean_std(data_dir, batch_size, num_workers, num_channels):

    # Step 1: Define which dataset to calculate on
    dataset = datasets.ImageFolder(data_dir + "train", transform=transforms.ToTensor(),)

    # Step 2: Create a DataLoader with for batched data loading
    full_loader = DataLoader(dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size, pin_memory=True)

    # Step 3: Initialize variables for mean and std
    mean = zeros(num_channels)
    std = zeros(num_channels)

    # Step 4: Calculate the mean and std for the dataset in parallel
    for inputs, _ in tqdm(full_loader, desc="==> Computing mean and std"):
        # Step 5: Compute mean and std for each channel using PyTorch operations
        mean += inputs.mean(dim=(0, 2, 3))
        std += inputs.std(dim=(0, 2, 3))

    # Step 6: Normalize the mean and std by dividing by the dataset size
    mean /= len(dataset)
    std /= len(dataset)

    print('MEAN', mean)
    print('STD', std)

    return mean, std

```

Running this for our training dataset w

#### Define ML model
#### Define hyperparamets
#### Train
#### Evaluate results
#### Hyperparameter tuning

