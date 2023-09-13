---
title: 'Two approaches for Kick Detection'
description: 'A guide to detect kick made in a football match'
pubDate: 'Sep 14 2023'
heroImage: '/with_ball_1.png'
---

## Description
This guide covers my attempts how to detect kicks in a football match, either by training a classification model or by using pretrained models to our advantage.

## Sections: 

- Objective
- Dataset
- Method 1. 
    - Preprocess data (normalize)
    - Split into train, val, test
    - Load data with data augmentation
    - Define model (simple dense NN)
    - Train
    - Evalute result
    - Iterate process. 
- Method 2. 
    - Pose estimation of players
    - Overlapping limbs

## Objective
The objective is to develop a solution capable of determining whether a kick is taking place in a given frame, with the expectation of a proof-of-concept solution.

## Kick dataset
Before we begin, let's take a closer look at the dataset's structure. The dataset comprises of a football match captured at a rate of one frame per second. The match is split into two folders: `with_ball` and `without_ball`, both of which contain YOLOv5 annotations for bounding boxes around the ball. I initially mistakenly assumed that these bounding boxes represented the players' legs when they kicked the ball, which led to inefficient use of my time. This serves as a reminder to always double-check and ensure a clear understanding of the dataset's contents.

Now, let's proceed to examine some examples from the dataset. Below, you'll find two images, one from the `with_ball` folder and the other from the `without_ball` folder:
##### 1. Image with ball
![example of photo with ball](/with_ball_2.png)

##### 2. Image without ball
![example of photo without ball](/without_ball_1.jpg)




As expected, the first photo contains a ball, while the second one does not. Excellent! Now, before we proceed, it's important to pause for a moment. Upon further inspection of the `without_ball` folder, I encountered some questionable images. Here are two examples that caught my attention:

##### 4. Example of images *"without"* ball
<div style="display: flex; justify-content: space-between;">
    <img src="/without_ball_3.jpg" alt="Kick Image 1" style="width: 47.5%;" />
    <img src="/without_ball_2.jpg" alt="Kick Image 2" style="width: 47.5%;" />
</div>

The first image is incorrectly labeled as it contains a ball, and the second image seems like an error. These examples do not mean the dataset is worthless and should be disregarded; it just means it has not been manually combed through to remove incorrect labels and ensure high quality.

With all of this in mind, let's explore what we've learned from the dataset. First, it's essential to emphasize the importance of double-checking and fully understanding your dataset. Second, not all datasets are equal in quality, and improving dataset quality can significantly boost performance. Lastly, we're still uncertain about how this dataset connects to our set objectives.

## First idea

The main issue with the dataset is that our objective is to detect when a kick has occurred, but no annotations have been provided to train a machine learning model. So what is the only logical solution? You guessed it! Annotate the data yourself.

To save time and keep things simple, I've decided to focus on the 'with_ball' folder and manually tag the images where players are kicking the ball. In that folder, there are 1550 images, but only 64 of them actually show players kicking the ball. I think this number would be much higher if we had more images, as many of the pictures are of players right before or after they kick the ball, not during the kick itself. Below, I've included two examples of images where I believe you can clearly see a kick happening.


##### 5. Examples of kicking the ball

<div style="display: flex; justify-content: space-between;">
    <img src="/kick_2.jpg" alt="Kick Image 2" style="width: 47.5%;" />
    <img src="/kick_1.jpg" alt="Kick Image 1" style="width: 47.5%;" />
</div>

