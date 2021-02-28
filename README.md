# "It is just a flu": Assessing the Effect of Watch History on YouTube's Pseudoscientific Video Recommendations

### Authors: Kostantinos Papadamou, Savvas Zannettou, Jeremy Blackburn, Emiliano De Cristofaro, Gianluca Stringhini, and Michael Sirivianos

## Abstract
The role played by YouTube's recommendation algorithm in unwittingly promoting misinformation and conspiracy theories is not entirely understood. 
Yet, this can have dire real-world consequences, especially when pseudoscientific content is promoted to users at critical times, such as the COVID-19 pandemic. 
In this paper, we set out to characterize and detect pseudoscientific misinformation on YouTube. 
We collect 6.6K videos related to COVID-19, the Flat Earth theory, as well as the anti-vaccination and anti-mask movements. Using crowdsourcing, we annotate them as pseudoscience, legitimate science, or irrelevant and train a deep learning classifier to detect pseudoscientific videos with an accuracy of 0.79.
We quantify user exposure to this content on various parts of the platform and how this exposure changes based on the user's watch history. 
We find that YouTube suggests more pseudoscientific content regarding traditional pseudoscientific topics (e.g., flat earth, anti-vaccination) than for emerging ones (like COVID-19). 
At the same time, these recommendations are more common on the search results page than on a user's homepage or when actively watching videos. 
Finally, we shed light on how a user's watch history substantially affects the type of recommended videos.

Preprint available <a href="https://arxiv.org/abs/2010.11638">here</a>.

If you make use of any modules available in this codebase in your work, please cite the following paper:
```latex
@article{papadamou2020just,
    title={'It is just a flu': Assessing the Effect of Watch History on YouTube's Pseudoscientific Video Recommendations},
    author={Papadamou, Kostantinos and Zannettou, Savvas and Blackburn, Jeremy and De Cristofaro, Emiliano and Stringhini, Gianluca and Sirivianos, Michael},
    journal={arXiv preprint arXiv:2010.11638},
    year={2020}
}
```

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Part 1: Detection of Pseudoscientific Videos](#part-1-detection-of-pseudoscientific-videos)
  - [Classifier Architecture](#11-classifier-architecture)
  - [Prerequisites](#12-prerequisites)
  - [Classifier Codebase (Model Training)](#13-training-the-classifier)
  - [Classifier Usage](#14-classifier-usage)
- [Part 2: YouTube Recommendation Algorithm Audit Framework](#part-2-youtube-recommendation-algorithm-audit-framework)
  - [Framework Prerequisites](#21-framework-prerequisites)
  - [User Profile Creation](#22-user-profile-creation)
  - [Framework Usage (Running Experiments)](#23-framework-usage)
  - [Framework Common Issues](#24-framework-common-issues)
- [Acknowledgements](#acknowledgements)
- [LICENSE](#license)


# Overview
YouTube has revolutionized the way people discover and consume video content online.
However, while YouTube facilitates easy access to hundreds of well-produced educational,entertaining, and trustworthy news videos, mistargeted and abhorrent content is also common.
While the scientific community has repeatedly pinpointed the need for effectively moderating inappropriate content, the various types of inappropriate content on the platform are relatively unstudied. 
At the same time, the role played by YouTube’s recommendation algorithm in unwittingly promoting such content is not entirely understood.
In our work, we study pseudoscientific misinformation on YouTube by focusing on quantifying user exposure to pseudoscientific misinformation on various parts of the platform, and how this exposure changes based on the user's watch history.
To do this, we first develop a deep learning model that detects pseudoscientific videos, as well as a methodology that allow us to simulate the behavior of logged-in and non-logged-in users with varying interests who casually browsing YouTube.

In this repository, we provide to the research community the source code of the developed classifier, as well as the source code of our methodology.
In particular, the ability to run this kind of experiments while taking into account users' viewing history will be beneficial to researchers focusing on demystifying YouTube’s recommendation algorithm—irrespective of the topic of interest. 
Our methodology and codebase are generic and can be used to study other topics besides pseudoscience, e.g., additional conspiracy theories.
More specifically, we make publicly available the following set of tools and libraries:
1. The codebase of a Deep Learning Classifier for pseudoscientific videos detection on YouTube;
2. The trained classifier and a library that simplifies the usage of the classifier and implements all the required tasks for the classification of YouTube videos;
3. An open source library that provides a unified framework for assessing the effects of personalization on YouTube video recommendations in multiple parts of the platform.

# Installation
Follow the steps below to install and configure all prerequisites for both the training and usage of the Pseudoscientific Content Detection Classifier (Part 1), and for using our YouTube Audit Framework (Part 2). 

### 1. Create a Python >=3.6 Virtual Environment
```bash
python3 -m venv virtualenv

source virtualenv/bin/activate
```

### 2. Install required packages
```bash
pip install -r requirements.txt
```

### 3. Install MongoDB
To store the metadata of YouTube videos, as well as for other information we use MongoDB. Install MongoDB on your own system or server using:

- **Ubuntu:** Follow instructions <a href="https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/">here</a>.
- **Mac OS X:** Follow instructions <a href="https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/">here</a>.

#### MongoDB Graphical User Interface 
We suggest the use of <a href="https://robomongo.org/">Robo3T</a> as a user interface for interacting with your MongoDB instance.


### 4. Install Additional Requirements

#### 4.1. Install the youtube-dl package
```bash
pip install youtube-dl
```
**Make use of ```youtube-dl``` wisely, carefully sending requests so that you do not spam YouTube with requests and get blocked.

#### 4.2. Install Google APIs Client library for Python
This is the library utilized to call the YouTube Data API from Python
```bash
pip install --upgrade google-api-python-client
```

### 5. HTTPS Proxies
Our codebase uses HTTPS Proxies for multiple purposes: 
- For downloading the transcripts of YouTube videos; and 
- The YouTube Recommendation Algorithm Audit Framework uses an HTTPS Proxy for each one of the user profiles and browser instances that it maintains. 
  This is mainly to ensure that all User Profiles used in our framework have the same geolocation and avoid changes to our results due to geolocation personalization.

You can either use your own HTTPS Proxies or buy some online and set them in the following files:
- ```youtubeauditframework/userprofiles/....py```: Includes the HTTPS Proxies used to simulate distinct logged-in user profiles accessing YouTube from specific geolocations. 
  Preferrably, according to our Audit framework, all HTTPS Proxies set in this file MUST be from similar locations (e.g., "US-San Fransisco-California"). 
- ```helpers/config/config.py```: Includes the HTTPS Proxies used to download the transcript of YouTube videos using ```youtube-dl```.



# Part 1: Detection of Pseudoscientific Videos
We implement a deep learning model geared to detect pseudoscientific YouTube videos. 
As also described in our paper, to train and test our model we use the dataset available <a href="">here</a>.

## 1.1. Classifier Architecture
![Model Architecture Diagram](https://github.com/kostantinos-papadamou/pseudoscience-paper/blob/master/classifier/architecture/model_architecture.png)

### Description
Our classifier consists of four different branches, each processing a distinct input feature type: snippet, video tags, transcript, and the top 200 comments of a video. 
Then, all four branches' outputs are concatenated to form a five-layer, fully-connected neural network that merges their output and drives the final classification. 
The classifier uses <a href="https://fasttext.cc/">fastText</a>, a library for efficient learning of word/document-level vector representations and sentence classification. 
We use fastText to generate vector representations (embeddings) for all the available video metadata in text.
For each input feature, we use the <a href="https://fasttext.cc/docs/en/english-vectors.html">pre-trained fastText models (1)</a> and fine-tune them using each of our input features.
These models extract a 300-dimensional vector representation for each of the following input features of our dataset:
- **Video Snippet:** Concatenation of the title and the description of the video.
- **Video Tags:** Words defined by the uploader of a video to describe the content of the video.
- **Transcript:** Naturally, this is one of the most important features, as it describes the video’s actual content. (It includes the subtitles uploaded by the creator of the video or auto-generated by YouTube.) 
  The classifier uses the fine-tuned model to learn a vector representation of the concatenated text of the transcript.
- **Comments:** We consider the top 200 comments of the video as returned by the YouTube Data API. 
  We first concatenate each video’s comments and use them to fine-tune the fastText model and extract vector representations.

The second part of the classifier ("Fusing Network") is essentially a four-layer, fully-connected, dense neural network. 
We use a Flatten utility layer to merge the outputs of the four branches of the first part of the classifier, creating a 1200-dimensional vector. 
This vector is processed by the four subsequent layers comprising 256, 128, 64, and 32 units, respectively, with ReLU activation. 
To avoid over-fitting, we regularize using the Dropout technique; at each fully-connected layer, we apply a Dropout level of d=0.5, i.e., during each iteration of training, half of each layer's units do not update their parameters. 
Finally, the Fusing Network output is fed to the last neural network of two units with softmax activation, which yields the probabilities that a particular video is pseudoscientific or not. 
We implement our classifier using Keras with Tensorflow as the back-end.

## 1.2. Prerequisites
### 1.2.1.  Download pre-trained fastText word vectors that we fine-tune during feature engineering on our dataset: 
```bash
cd pseudoscientificvideosdetection/models/feature_extraction

wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip

unzip wiki-news-300d-1M.vec.zip
```

### 1.2.2. Create the following MongoDB Collections:
- ```groundtruth_videos```
- ```groundtruth_videos_comments```
- ```groundtruth_videos_transcripts```

** If you are using our <a href="https://zenodo.org/record/4558469#.YDfBCmr7Rqs">dataset</a> please make sure that you create the appropriate MongoDB collections and import the data in each collection. 



## 1.3. Training the Classifier
Below we describe how can you use the codebase to train our classifier using either our own data available <a href="">here</a>, 
or using your own dataset. 
We note that, our model is optimized for the detection of pseudoscientific content related to COVID-19, Anti-vaccination, Anti-mask, and Flat Earth. 
However, feel free to extend/enhance the provided codebase implementing your own deep learning model optimized for your use case.

### Step A. Fine-tune separate fastText models for each Video Metadata Type
In this step, we fine-tune four separate fastText models, one for each different video metadata type, 
which we use during the training of Deep Learning model to generate embeddings for each different video metadata type. 
This step is only required to run once.

### Step B. Train the Pseudoscience Deep Learning Model
At this step, we train and validate the Pseudoscientific Content Detection Deep Learning model using 10-fold cross-validation.
At the end of the training, the best model will be stored in: ```pseudoscientificvideosdetection\models\pseudoscience_model_final.hdf5```.

### Classifier Training Example:
```python
from dataset.DatasetUtils import DatasetUtils
from classifier.featureengineering.FeatureEngineeringModels import FeatureEngineeringModels
from classifier.training.ClassifierTraining import ClassifierTraining

# Create Objects
dataset = DatasetUtils()
featureEngineeringModels = FeatureEngineeringModels(dataset_object=dataset)

""" Step 1: Fine-tune separate fastText models for each Video Metadata Type """
# Video Snippet
# Generate Video Snippet fastText input features
featureEngineeringModels.prepare_fasttext_data(model_type='video_snippet')
# Fine-tune a fastText model for Video Snippet
featureEngineeringModels.finetune_model(model_type='video_snippet')

# Video Tags
# Generate Video Tags fastText input features
featureEngineeringModels.prepare_fasttext_data(model_type='video_tags')
# Fine-tune a fastText model for Video Tags
featureEngineeringModels.finetune_model(model_type='video_tags')

# Video Transcript
# Generate Video Transcript fastText input features
featureEngineeringModels.prepare_fasttext_data(model_type='video_transcript')
# Fine-tune a fastText model for Video Transcript
featureEngineeringModels.finetune_model(model_type='video_transcript')

# Video Comments
# Generate Video Comments fastText input features
featureEngineeringModels.prepare_fasttext_data(model_type='video_comments')
# Fine-tune a fastText model for Video Comments
featureEngineeringModels.finetune_model(model_type='video_comments')

""" Step 2: Train the Pseudoscience Deep Learning Model """
# Create a Classifier Training Object
classifierTrainingObject = ClassifierTraining(dataset_object=dataset)
# Train the Classifier
classifierTrainingObject.train_model()
```


## 1.4. Classifier Usage
In this repository, we also include a Python package that uses the trained classifier, which can be used by anyone who wants to train our classifier using our codebase and then use it to detect pseudoscientific videos on YouTube.
The available Python module accepts a YouTube Video ID as input, and implements all the necessary steps to download the required information of the given video, extract the required features, and classifies the video using the trained classifier.
Note that, the classifier is supposed to be trained and used to detect pseudoscientific videos on YouTube related to the following topics: a) COVID-19; b) Anti-vaccination; c) Anti-mask; and d) Flat Earth.

Finally, in order to use this package you first need to train the classifier using our classifier' codebase, and also to provide your own YouTube Data API key.
If case you want to train the classifier using our own dataset, you request access to it and download it from <a href="https://zenodo.org/record/4558469#.YDfBCmr7Rqs">here</a>.

### Classifier Usage Example:
```python
# Import the PSeudoscientific Videos Detection package
from pseudoscientificvideosdetection.PseudoscienceClassifier import PseudoscienceClassifier
from helpers.YouTubeVideoDownloader import YouTubeVideoDownloader

# Create an object of the classifier
pseudoscienceClassifier = PseudoscienceClassifier()
ytDownloader = YouTubeVideoDownloader(api_key='<YOUR_YOUTUBE_DATA_API_KEY>')

# Download YouTube Video Details
# Note: You can replace the following line with your own method that creates the
#       dictionary with the video details. See YouTubeVideoDownloader class to
#       understand the format in which we convert the comments and the transcript
#       of the video before we pass them to the classifier
video_details = ytDownloader.download_video(video_id='<YOUTUBE_VIDEO_ID>')

# Make a prediction
prediction, confidence_score = pseudoscienceClassifier.classify(video_details=video_details) 
```



# Part 2: YouTube Recommendation Algorithm Audit Framework

## 2.1. Framework Prerequisites

### 2.1.1. Create the following MongoDB Collections:

- ```audit_framework_videos```: All videos of the YouTube Audit framework will be stored in this collection.
- ```audit_framework_youtube_homepage```: Holds the details of each repetition of the **YouTube Homepage** audit experiment.
- ```audit_framework_youtube_search```: Holds the details of each repetition of the **YouTube Search Results** audit experiment.
- ```audit_framework_youtube_video_recommendations```: Holds the details of each repetition of the **YouTube Video Recommendations** audit experiment.

### 2.1.2. Download Google ChromeDriver
You can download the ChromeDriver that matches the one of your local Google Chrome version from <a href="https://chromedriver.chromium.org/downloads">here</a>. 
We recommend the usage of Google ```ChromeDriver 83.0.4103.39```, however, if you download another one please make sure that you update ```youtubeauditframework/utils/config.py```.

- **Linux:**
  ```bash
  cd youtubeauditframework/utils/webdrivers
  
  wget https://chromedriver.storage.googleapis.com/83.0.4103.39/chromedriver_linux64.zip
  
  unzip chromedriver_linux64.zip
  
  rm chromedriver_linux64.zip
  ```

- **MacOS X:**
  ```bash
  cd youtubeauditframework/utils/webdrivers
  
  wget https://chromedriver.storage.googleapis.com/83.0.4103.39/chromedriver_mac64.zip
  
  unzip chromedriver_mac64.zip
  
  rm chromedriver_mac64.zip
  ```


## 2.2. User Profile Creation

### 2.2.1. Create Google (YouTube) Accounts

1. create profile folders and login and train them

### 2.2.2. User Profile Training (Build User's Watch History)


## 2.3. Framework Usage
- login everytime from before

### 2.3.1. Running Experiments

### 2.3.2 Experiments Results Analysis


## 2.4. Framework Common Issues
Unfortunately, due to regular Google Chrome Updates or other updates on the YouTube Website, our framework may not function properly from time to time. 
In this case, we list below some of the most common issues that we faced to assist you with overcoming them when using our framework. 

#### - Google Chrome version and User-Agent of our crawler:
If you find problems running our framework (i.e., the browser is closing right after you start an experiment), then this probably due to a mismatch 
between the declared ChromeDriver downloaded, the User-Agent declared in ```youtubeauditframework/utils/config.py``` and the current version of your 
installed Google Chrome. It is better if all these three match and you can start by finding the version of your installed browser in its "About Google Chrome" section.
You can download the ChromeDriver that matches your installed Google Chrome and Operating System from <a href="https://chromedriver.chromium.org/downloads">here</a>.
Last, if you do not use the recommended ChromeDriver version then make sure that you update USer-Agent string in ```youtubeauditframework/utils/config.py```.

#### - Regular updates of YouTube's HTML/CSS codebase:
Automated functionalities of framework, like getting the recommended videos of a given video, or deleting the watching history 
of a logged-in user may not work from time to time and this is mainly because YouTube regularly updates its HTML and CSS classes. 
Hence, when you have such problems you may need to update the codebase of our framework with the latest XPaths of each element (i.e., button, video thumbnail) 
that you can find by inspecting each element on the YouTube website (using Google Chrome inspect option).    

#### - Ensure that all User Profiles are logged-in:
Before running an experiment with a given User Profile, ensure that this user is logged-in. 
You can do this by running the helper script in ```youtubeauditframework/userprofiles/helpers/manual_user_profile_login.py```.
You can set the desired User Profile in the beginning of this script, and when running this helper script the corresponding browser (with the details and activity of this user) will open. 
Then you will be able to manually follow the authentication  flow and authenticate this user profile on YouTube. 
Upon successful authentication, please ensure that you properly close the opened browser window before running any experiments using this user profile.




# Acknowledgements
Please see the <a href="https://arxiv.org/abs/2010.11638">paper</a> for funding details and non-code related acknowledgements.

# LICENSE

MIT License