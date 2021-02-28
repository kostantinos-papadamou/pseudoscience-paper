#!/usr/bin/python

import os
import fasttext
import multiprocessing


class FeatureEngineeringModels(object):
    """
    Class that fine-tunes and stores a fastText model for each Video metadata type
    using pre-trained word vectors available here: https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip

    The trained fastText models are then used during training and inference to generate
    vector representations (embeddings) for each available video metadata in text.
    """
    def __init__(self, dataset_object):
        # Set Base Directories
        self.DATA_DIR = 'dataset/data/feature_engineering_models_data'
        self.FEATURE_ENGINEERING_MODELS_DIR = 'pseudoscientificvideosdetection/models/feature_extraction'

        # Create a Dataset Object
        # self.DATASET = DatasetUtils()
        self.DATASET = dataset_object
        return

    def prepare_fasttext_data(self, model_type, overwrite=False):
        """
        Method that prepares the input features and stores them in a filename that will
        be used later to fine-tune the fastText model
        :param video_metadata_type: one of 'video_snippet', 'video_tags', 'video_transcript', or 'video_comments'
        :param input_data: preprocessed data of the given Video metadata type
        :param overwrite: whether to overwrite existing saved features (if exists)
        :return:
        """
        # Get input features
        if model_type == 'video_snippet':
            input_features = self.DATASET.get_video_snippet_features()
        elif model_type == 'video_tags':
            input_features = self.DATASET.get_video_tags_features()
        elif model_type == 'video_transcript':
            input_features = self.DATASET.get_video_transcript_features()
        elif model_type == 'video_comments':
            input_features = self.DATASET.get_video_comments_features()

        # Convert into features to fastText input data
        fasttext_input_filename = '{0}/{1}_train_data.txt'.format(self.DATA_DIR, model_type)
        if not overwrite and os.path.isfile(fasttext_input_filename):
            return
        with open(fasttext_input_filename, mode='w') as file:
            for row in input_features:
                if len(row) > 0:
                    file.write('{0}\n'.format(row))
        return

    def finetune_model(self, model_type, overwrite=False):
        """
        Method that trains an unsupervised fasttext model on our dataset for the given
        Video metadata type and stores is so that it can be used during the training of
        the Pseudoscience Classifier for extracting the embeddings from the input features
        :param model_type: 'video_snippet', 'video_tags', 'video_transcript', or 'video_comments'
        :param overwrite: whether to retrain and overwrite existing saved fastText model (if exists)
        :return:
        """
        # Create fastText input data filename
        fasttext_model_filename = '{0}/fasttext_model_{1}.bin'.format(self.FEATURE_ENGINEERING_MODELS_DIR, model_type)
        if not os.path.isfile(fasttext_model_filename) or overwrite:
            # Train unspervised fastText model
            model = fasttext.train_unsupervised(input='{0}/{1}_train_data.txt'.format(self.DATA_DIR, model_type),
                                                pretrainedVectors='wiki-news-300d-1M.vec',
                                                dim=300,
                                                minn=2,
                                                maxn=5,
                                                thread=multiprocessing.cpu_count() - 1,  # run in multiple cores
                                                verbose=2)
            # Save trained model
            model.save_model(fasttext_model_filename)
        return
