#!/usr/bin/python

# import sys
# sys.path.insert(0, '.')
from DatasetUtils import DatasetUtils
import fasttext
import numpy as np
import os
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from joblib import Parallel, delayed


class DataPreparation(object):
    """
    Class that prepares all type of input features required for the training of our
    Pseudoscience Videos Detection Classifier
    """
    def __init__(self, dataset_object):
        # Set Base Directories
        self.INPUT_FEATURES_DIR = 'dataset/data/input_features'
        self.FEATURE_ENGINEERING_MODELS_DIR = 'pseudoscientificvideosdetection/models/feature_extraction'

        # Get Dataset Object
        self.DATASET = dataset_object
        return

    def get_video_snippet_model_input_features(self, overwrite=False):
        """
        Method that generates an Embedding Vector of each video Snippet (video title + video description)
        :param overwrite: True if existing pre-generated features should be re-generated, False if not
        :return:
        """
        # Create Video Snippet Features filename
        video_snippet_features_filename = '{0}/video_snippet_sentence_embeddings.p'.format(self.INPUT_FEATURES_DIR)

        print('\n--- Getting VIDEO SNIPPET features sentence embeddings using fine-tuned fastText model')
        if not os.path.isfile(video_snippet_features_filename) or overwrite:
            # Get Video Snippets raw preprocessed
            video_snippets_raw = self.DATASET.get_video_snippet_features()

            # Load fine-tuned fastText Model
            video_snippet_fasttext_model = fasttext.load_model(path='{0}/fasttext_model_video_snippet.bin'.format(self.FEATURE_ENGINEERING_MODELS_DIR))

            # Generate a single Embedding Vector for each Video Snippet
            video_snippet_features = list()
            for video_snippet in video_snippets_raw:
                video_snippet_features.append(video_snippet_fasttext_model.get_sentence_vector(text=video_snippet))

            # Save Video Snippet sentence-level embeddings
            pickle.dump(video_snippet_features, open(video_snippet_features_filename, mode='wb'))
            # Free some memory
            del video_snippet_fasttext_model
        else:
            # Return pre-generated input features
            video_snippet_features = pickle.load(open(video_snippet_features_filename, mode='rb'))
        return video_snippet_features

    def get_video_tags_model_input_features(self, overwrite=False):
        """
        Method that generates an Embedding Vector of each Video's Tags
        :param overwrite: True if existing pre-generated features should be re-generated, False if not
        :return:
        """
        # Create Video Tags Features filename
        video_tags_features_filename = '{0}/video_tags_sentence_embeddings.p'.format(self.INPUT_FEATURES_DIR)

        print('\n--- Getting VIDEO TAGS features sentence embeddings using fine-tuned fastText model')
        if not os.path.isfile(video_tags_features_filename) or overwrite:
            # Get Video Snippets raw preprocessed
            video_tags_raw = self.DATASET.get_video_tags_features()

            # Load fine-tuned fastText Model
            video_tags_fasttext_model = fasttext.load_model(path='{0}/fasttext_model_video_tags.bin'.format(self.FEATURE_ENGINEERING_MODELS_DIR))

            # Generate a single Embedding Vector for each Video's Tags
            video_tags_features = list()
            for video_tags in video_tags_raw:
                video_tags_features.append(video_tags_fasttext_model.get_sentence_vector(text=video_tags))

            # Save Video Tags sentence-level embeddings
            pickle.dump(video_tags_features, open(video_tags_features_filename, mode='wb'))
            # Free some memory
            del video_tags_fasttext_model
        else:
            # Return pre-generated input features
            video_tags_features = pickle.load(open(video_tags_features_filename, mode='rb'))
        return video_tags_features

    def get_video_transcript_model_input_features(self, overwrite=False):
        """
        Method that generates an Embedding Vector of each video's Transcript
        :param overwrite: True if existing pre-generated features should be re-generated, False if not
        :return:
        """
        # Create Video Transcripts Features filename
        video_transcript_features_filename = '{0}/video_transcripts_sentence_embeddings.p'.format(self.INPUT_FEATURES_DIR)

        print('\n--- Getting VIDEO TRANSCRIPT features sentence embeddings using fine-tuned fastText model')
        if not os.path.isfile(video_transcript_features_filename) or overwrite:
            # Get Video Transcript raw preprocessed
            video_transcripts_raw = self.DATASET.get_video_transcript_features()

            # Load fine-tuned fastText Model
            video_transcript_fasttext_model = fasttext.load_model(path='{0}/model_unsupervised_video_transcript.bin'.format(self.FEATURE_ENGINEERING_MODELS_DIR))

            # Generate a single Embedding Vector for each Video Transcript
            video_transcript_features = list()
            for video_transcript in video_transcripts_raw:
                video_transcript_features.append(video_transcript_fasttext_model.get_sentence_vector(text=video_transcript))

            # Save Video Transcript sentence-level embeddings
            pickle.dump(video_transcript_features, open(video_transcript_features_filename, mode='wb'))
            # Free some memory
            del video_transcript_fasttext_model
        else:
            # Return pre-generated input features
            video_transcript_features = pickle.load(open(video_transcript_features_filename, mode='rb'))
        return video_transcript_features

    def get_video_comments_model_input_features(self, overwrite=False):
        """
        Method that generates an Embedding Vector of each Video's Comments
        :param overwrite: True if existing pre-generated features should be re-generated, False if not
        :return:
        """
        # Create Video Comments Features filename
        video_comments_features_filename = '{0}/video_comments_merged_sentence_embeddings.p'.format(self.INPUT_FEATURES_DIR)

        print('\n--- Getting VIDEO COMMENTS features and Embedding matrix using fine-tuned fastText model')
        if not os.path.isfile(video_comments_features_filename) or overwrite:
            # Get Video Comments raw preprocessed
            video_comments_raw = self.DATASET.get_video_comments_features()

            # Merge the comments of each video
            final_video_comments_raw = [' '.join(video_comments) for video_comments in video_comments_raw]

            # Load fine-tuned fastText Model
            video_comments_fasttext_model = fasttext.load_model(path='{0}/model_unsupervised_video_comments.bin'.format(self.FEATURE_ENGINEERING_MODELS_DIR))

            # Generate a single Embedding Vector for each Video Comments
            video_comments_features = [video_comments_fasttext_model.get_sentence_vector(text=video_comments_merged) for video_comments_merged in final_video_comments_raw]

            # Save Video Comments sentence-level embeddings
            pickle.dump(video_comments_features, open(video_comments_features_filename, mode='wb'))
            # Free some memory
            del video_comments_fasttext_model
        else:
            # Return pre-generated input features
            video_comments_features = pickle.load(open(video_comments_features_filename, mode='rb'))
        return video_comments_features
