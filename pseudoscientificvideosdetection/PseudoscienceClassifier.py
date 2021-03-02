#!/usr/bin/python

import os
import numpy as np
from decimal import Decimal
from tensorflow.keras.models import load_model
from dataset.DatasetUtils import DatasetUtils
import fasttext


class PseudoscienceClassifier(object):
    """
    Class that implements all the necessary methods in order to classify a given Video
    into Pseudoscience and Science categories
    """
    def __init__(self, classification_threshold=0.7, gpu_training=False):
        """
        Constructor
        :param classification_threshold: the preferred classification threshold (default: 0.7)
        :param gpu_training: True if training on GPU, False if training on CPU
        """
        # Enable GPU Training
        if gpu_training:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Initialize Variables
        self.NB_CLASSES = 2
        self.CLASSES = ['science', 'pseudoscience']

        # Create a Dataset Object
        self.DATASET = DatasetUtils()

        # Set the base directory of the FastText Classifiers
        self.FASTTEXT_MODELS_DIR = 'pseudoscientificvideosdetection/models/feature_extraction'
        # Load FastText Classifiers
        if not os.path.isfile('{0}/model_unsupervised_video_snippet.bin'.format(self.FASTTEXT_MODELS_DIR)):
            exit('Cannot find fasttext feature extractor for VIDEO SNIPPET')
        self.FASTTEXT_VIDEO_SNIPPET = fasttext.load_model(path='{0}/model_unsupervised_video_snippet.bin'.format(self.FASTTEXT_MODELS_DIR))
        if not os.path.isfile('{0}/model_unsupervised_video_tags.bin'.format(self.FASTTEXT_MODELS_DIR)):
            exit('Cannot find fasttext feature extractor for VIDEO TAGS')
        self.FASTTEXT_VIDEO_TAGS = fasttext.load_model(path='{0}/model_unsupervised_video_tags.bin'.format(self.FASTTEXT_MODELS_DIR))
        if not os.path.isfile('{0}/model_unsupervised_video_transcript.bin'.format(self.FASTTEXT_MODELS_DIR)):
            exit('Cannot find fasttext feature extractor for VIDEO TRANSCRIPT')
        self.FASTTEXT_VIDEO_TRANSCRIPT = fasttext.load_model(path='{0}/model_unsupervised_video_transcript.bin'.format(self.FASTTEXT_MODELS_DIR))
        if not os.path.isfile('{0}/model_unsupervised_video_comments.bin'.format(self.FASTTEXT_MODELS_DIR)):
            exit('Cannot find fasttext feature extractor for VIDEO COMMENTS')
        self.FASTTEXT_VIDEO_COMMENTS = fasttext.load_model(path='{0}/model_unsupervised_video_comments.bin'.format(self.FASTTEXT_MODELS_DIR))

        # Load the Pseudoscience Classifier
        self.pseudoscience_model_filename = 'pseudoscientificvideosdetection/models/pseudoscience_model_final.hdf5'
        if not os.path.isfile(self.pseudoscience_model_filename):
            exit('Cannot find a trained Pseudoscience Classifier')
        self.PSEUDOSCIENCE_CLASSIFIER = load_model(self.pseudoscience_model_filename)

        # Read Classification Threshold (default: 0.7)
        self.CLASSIFICATION_THRESHOLD = classification_threshold
        return

    def __del__(self):
        """
        Destructor
        Ensure that you free some memory by deleting all the loaded models
        :return:
        """
        del self.FASTTEXT_VIDEO_SNIPPET
        del self.FASTTEXT_VIDEO_TAGS
        del self.FASTTEXT_VIDEO_TRANSCRIPT
        del self.FASTTEXT_VIDEO_COMMENTS
        del self.PSEUDOSCIENCE_CLASSIFIER
        return

    @staticmethod
    def key_exists(element, *keys):
        """
        Check if *keys (nested) exists in `element` (dict).
        :param keys:
        :return: True if key exists, False if not
        """
        if type(element) is not dict:
            raise AttributeError('keys_exists() expects dict as first argument.')
        if len(keys) == 0:
            raise AttributeError('keys_exists() expects at least two arguments, one given.')

        _element = element
        for key in keys:
            try:
                _element = _element[key]
            except KeyError:
                return False
        return True

    def classify(self, video_details):
        """
        Method that receives the information of a given YouTube Video and classifies it as Science or Pseudoscience
        :param video_id:
        :return:
        """

        """ Prepare Classifier Input """
        # --- VIDEO SNIPPET
        video_snippet = '{} {}'.format(video_details['snippet']['title'], video_details['snippet']['description'])
        # Preprocess Video Snippet
        video_snippet = self.DATASET.preprocess_text(text=video_snippet)
        # Get Embedding
        X_video_snippet = self.FASTTEXT_VIDEO_SNIPPET.get_sentence_vector(text=video_snippet)

        # --- VIDEO TAGS
        video_tags = ""
        if self.key_exists(video_details, 'tags', 'tags') and len(video_details['snippet']['tags']) > 0:
            video_tags = ' '.join(video_details['snippet']['tags'])
        # Preprocess Video Tags
        if video_tags != "":
            video_tags = self.DATASET.preprocess_text(text=video_tags)
        # Get Embedding
        X_video_tags = self.FASTTEXT_VIDEO_TAGS.get_sentence_vector(text=video_tags)

        # --- VIDEO TRANSCRIPT
        video_transcript = self.DATASET.read_video_transcript(video_id=video_details['id'])
        video_transcript_processed = self.DATASET.preprocess_video_transcript(video_captions=video_transcript)
        # Get Embedding
        X_video_transcript = self.FASTTEXT_VIDEO_TRANSCRIPT.get_sentence_vector(text=video_transcript_processed)

        # --- VIDEO COMMENTS
        video_comments = self.DATASET.read_video_comments(video_id=video_details['id'])
        video_comments_preprocessed = self.DATASET.preprocess_video_comments(video_comments=video_comments)
        # Get Embedding
        X_video_comments = self.FASTTEXT_VIDEO_COMMENTS.get_sentence_vector(text=' '.join(video_comments_preprocessed))

        """ Classify Video """
        # Create Classifier Input
        classifier_input = [[X_video_snippet], [X_video_tags], [X_video_transcript], [X_video_comments]]

        # Perform Classification
        predicted_proba = self.PSEUDOSCIENCE_CLASSIFIER.predict(classifier_input, batch_size=1)

        # Decode Predicted probability and convert it to a Label
        if self.CLASSIFICATION_THRESHOLD is not None:

            pseudoscience_probability = np.round(predicted_proba.item(1), decimals=3)
            # Check the Pseudoscience Probability against the Classification Threshold
            if Decimal(pseudoscience_probability) >= Decimal(self.CLASSIFICATION_THRESHOLD):
                # PSEUDOSCIENCE is the correct class
                predicted_class = self.CLASSES[1]
            else:
                predicted_class = self.CLASSES[0]
        else:
            # Convert probability to class offset
            prediction = predicted_proba.argmax(axis=-1)
            # Get predicted class from offset
            predicted_class = self.CLASSES[prediction[0]]

        # Convert probabilities to float
        science_proba = np.round(predicted_proba.item(0), decimals=3)
        pseudoscience_proba = np.round(predicted_proba.item(1), decimals=3)
        # print('--- [VIDEO_ID: {}] PREDICTED CLASS: {}, PROBAS: [{}, {}]\n'.format(video_details['id'], predicted_class.upper(), science_proba, pseudoscience_proba))

        # Find the appropriate Confidence Score
        conf_score = science_proba if predicted_class == 'science' else pseudoscience_proba
        return predicted_class, conf_score
