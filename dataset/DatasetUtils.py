#!/usr/bin/python

from pymongo import MongoClient
import json
import os
import re
import unicodedata
import string
import numpy as np
import pickle
import itertools
from tqdm import tqdm
from keras.utils import to_categorical

import contractions
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import nltk
stop_words = set(nltk.corpus.stopwords.words('english'))
stop_words.remove('no')
stop_words.remove('not')
from nltk.stem import WordNetLemmatizer
lematizer = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
porterStemmer = PorterStemmer()


class DatasetUtils(object):
    """
    Class that contains all the methods for the processing and preprocessing of our dataset
    """
    def __init__(self):
        #
        # MongoDB Configuration
        #
        # Host and Port
        self.client = MongoClient('localhost', 27017)
        # DB name
        self.db = self.client['youtube_pseudoscience_dataset']
        # Collection name
        self.groundtruth_videos_col = self.db.groundtruth_videos
        self.groundtruth_videos_comments_col = self.db.groundtruth_videos_comments
        self.groundtruth_videos_transcripts_col = self.db.groundtruth_videos_transcripts

        # Video Metadata Base Directories
        self.VIDEO_TRANSCRIPT_BASE_DIR = 'videosdata/transcript'
        self.VIDEO_COMMENTS_BASE_DIR = 'videosdata/comments'

        # Input Features filenames
        self.VIDEO_SNIPPET_FEATURES_FILENAME = 'dataset/data/video_snippet_features.p'
        self.VIDEO_TAGS_FEATURES_FILENAME = 'dataset/data/video_tags_features.p'
        self.VIDEO_TRANSCRIPT_FEATURES_FILENAME = 'dataset/data/video_transcript_features.p'
        self.VIDEO_COMMENTS_FEATURES_FILENAME = 'dataset/data/video_comments_features.p'

        # CLASSES
        self.classes = ['science', 'pseudoscience']

        # Get Ground-truth Videos
        self.GROUNDTRUTH_VIDEOS = self.get_groundtruth_videos()
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

    @staticmethod
    def create_pickle_file_and_write_data(filename, data, protocol):
        """
        Method that receives as input the absolute path of a file and some data (numpy array) and
        then it uses pickle to dump the provided data into that file
        """
        # Write the provided data (bytes) in the created file
        pickle.dump(data, open(filename, 'wb'), protocol=protocol)
        return

    def get_groundtruth_videos(self):
        """
        Method that returns the ids of the Ground-truth videos
        :return:
        """
        groundtruth_videos = self.groundtruth_videos_col.find({}, {'id': 1})
        return [video['id'] for video in groundtruth_videos]

    def get_groundtruth_labels(self):
        """
        Method that returns the labels of our Ground Truth Videos
        :return:
        """
        groundtruth_videos = self.groundtruth_videos_col.find({}, {'id': 1, 'classification': 1})

        video_labels = list()
        science_videos = 0
        pseudoscience_videos = 0
        irrelevant_videos = 0
        for video in groundtruth_videos:
            # Find the correct label
            if video['classification']['classification_category'] == 'science':
                science_videos += 1
            elif video['classification']['classification_category'] == 'pseudoscience':
                pseudoscience_videos += 1
            elif video['classification']['classification_category'] == 'irrelevant':
                irrelevant_videos += 1

            # Add Video details
            if video['classification']['classification_category'] == 'irrelevant':
                video_labels.append('science')
            else:
                video_labels.append(video['classification']['classification_category'])
        print('\n\n--- [GROUND TRUTH VIDEOS] SCIENCE: {} | PSEUDOSCIENCE: {}'.format(science_videos+irrelevant_videos, pseudoscience_videos))
        return video_labels

    def get_groundtruth_labels_one_hot_encoded(self, perform_one_hot=True):
        """
        Method that returns a list with all the ground truth labels one-hot encoded
        :param perform_one_hot:
        :return:
        """
        groundtruth_videos = self.groundtruth_videos_col.find({}, {'id': 1, 'classification': 1})
        ground_truth_labels = [video['classification']['classification_category'] for video in groundtruth_videos]

        final_groundtruth_labels = list()
        for label in ground_truth_labels:
            if label == 'irrelevant':
                label = 'science'
            final_groundtruth_labels.append(label)

        # Init variables
        groundtruth_videos_labels_encoded = list()
        # Perform one-hot categorical encoding on each label and append it to the result
        for label in final_groundtruth_labels:
            if perform_one_hot:
                groundtruth_videos_labels_encoded.append(self.get_class_one_hot(label))
            else:
                groundtruth_videos_labels_encoded.append(self.get_class_to_categorical(label))
        return np.array(groundtruth_videos_labels_encoded)

    def get_class_one_hot(self, class_str):
        """
        Method that given a class as a string, return its one-hot encoded in the classes list
        :param class_str:
        :return:
        """
        # Now one-hot it. e.g., to_categorical(inappropriate = 2) => [0, 0, 1, 0]
        label_hot = to_categorical(self.classes.index(class_str), len(self.classes))

        assert len(label_hot) == len(self.classes)
        return label_hot

    def get_class_to_categorical(self, class_str):
        """
        Method that given a class as a string, return its number in the classes list
        :param class_str:
        :return:
        """
        label_to_categorical = self.classes.index(class_str)
        return label_to_categorical

    @staticmethod
    def strip_html_tags(text):
        """ Remove HTML tags from text """
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text

    @staticmethod
    def remove_URL_linking(text):
        """ Remove URLs from a text """
        text = re.sub('https?://[A-Za-z0-9./]+', '', text)
        text = re.sub(r"http\S+", "", text)
        return text

    @staticmethod
    def remove_accented_characters(text):
        """ Remove accented and non-ascii characters. For example convert é to e. """
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    @staticmethod
    def replace_contractions(text):
        """Replace contractions in string of text"""
        return contractions.fix(text)

    @staticmethod
    def remove_special_characters(text, remove_digits=False):
        """Remove special characters from text"""
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
        return text

    @staticmethod
    def remove_stopwords(tokens):
        """Remove stopwords from text (such as 'a', 'an', etc.)"""
        filtered_tokens = [token for token in tokens if token not in stop_words]
        return filtered_tokens

    @staticmethod
    def remove_multiple_spaces_newlines(text):
        # Substituting multiple spaces with single space
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        # remove extra newlines
        text = re.sub(r'[\r|\n|\r\n]+', ' ', text)
        return text

    @staticmethod
    def lematize_tokens(tokens):
        """Lematize all words in the given list of words"""
        tokens = [lematizer.lemmatize(word) for word in tokens]
        return tokens

    def preprocess_text(self, text):
        """Method that preprocess text for the FastText classifier"""

        # Substituting multiple spaces with single space
        text = self.remove_multiple_spaces_newlines(text=text)

        # Strip HTML Tags
        text = self.strip_html_tags(text=text)

        # Remove URLs
        text = self.remove_URL_linking(text=text)

        # Remove all special characters and digits
        text = self.remove_special_characters(text=text, remove_digits=True)

        # Remove accented characters (non-ascii)
        text = self.remove_accented_characters(text=text)

        # Replace Contractions (don't => do not)
        text = self.replace_contractions(text=text)

        # Remove Panctuations
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Lowercase
        text = text.strip().lower()

        # Lemmatize
        tokens = word_tokenize(text=text)
        tokens = self.lematize_tokens(tokens=tokens)

        # Remove stop words
        tokens = self.remove_stopwords(tokens=tokens)

        # Discard terms with less than min_chars
        min_chars = 3
        tokens = [token for token in tokens if len(token) >= min_chars]

        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    def preprocess_video_snippet(self, video_id):
        """
        Method that concatenates the Title + Description of the Video, and preprocess them
        before it returns them
        :param video_id:
        :return:
        """
        # Get Video details
        video_details = self.groundtruth_videos_col.find_one({'id': video_id}, {'snippet': 1})
        # Concatenate the different video metadata
        video_snippet = '{} {}'.format(video_details['snippet']['title'], video_details['snippet']['description'])
        video_snippet = self.preprocess_text(text=video_snippet)
        return video_snippet

    def get_video_snippet_features(self):
        """
        Method that gets the Video Snippet features for all all_videos in our Ground Truth
        :return:
        """
        all_video_snippet_features = list()
        if not os.path.isfile(self.VIDEO_SNIPPET_FEATURES_FILENAME):
            progress = tqdm(total=len(self.GROUNDTRUTH_VIDEOS))
            for video_id in self.GROUNDTRUTH_VIDEOS:
                all_video_snippet_features.append(self.preprocess_video_snippet(video_id=video_id))
                progress.update(1)
            progress.close()

            # Save them to file
            pickle.dump(all_video_snippet_features, open(self.VIDEO_SNIPPET_FEATURES_FILENAME, mode='wb'))
        else:
            # Read saved features from file
            all_video_snippet_features = pickle.load(open(self.VIDEO_SNIPPET_FEATURES_FILENAME, mode='rb'))

        return all_video_snippet_features

    def preprocess_video_tags(self, video_id):
        """
        Method that retrieves and preprocesses the Video Tags of a given YouTube Video
        :param video_id:
        :return:
        """
        # Get Video details
        video_details = self.groundtruth_videos_col.find_one({'id': video_id}, {'snippet': 1})

        # Read Video Tags if exist
        video_tags = ''
        if self.key_exists(video_details, 'snippet', 'tags'):
            video_tags = ' '.join(video_details['snippet']['tags'])

        # Preprocess the Video Tags before returning them
        video_tags_preprocessed = self.preprocess_text(text=video_tags)
        return video_tags_preprocessed

    def get_video_tags_features(self):
        """
        Method that return s the Video Tags features of all the Groundtruth videos
        :return:
        """
        all_video_tags_features = list()
        if not os.path.isfile(self.VIDEO_TAGS_FEATURES_FILENAME):
            progress = tqdm(total=len(self.GROUNDTRUTH_VIDEOS))
            for video_id in self.GROUNDTRUTH_VIDEOS:
                all_video_tags_features.append(self.preprocess_video_tags(video_id=video_id))
                progress.update(1)
            progress.close()

            # Save them to file
            pickle.dump(all_video_tags_features, open(self.VIDEO_TAGS_FEATURES_FILENAME, mode='wb'))
        else:
            # Read saved features from file
            all_video_tags_features = pickle.load(open(self.VIDEO_TAGS_FEATURES_FILENAME, mode='rb'))

        return all_video_tags_features

    def read_video_transcript(self, video_id):
        """
        Method that returns a list with all the captions of a given video read from a file
        :param video_id: the ID of the video to get its comments
        :return: a list of all the captions of the given video
        """
        video_transcript_parsed = list()
        video_transcript_filename = '{}{}/{}.en.vtt'.format(self.VIDEO_TRANSCRIPT_BASE_DIR, video_id[:3], video_id)
        if os.path.isfile(video_transcript_filename):
            transcript_file = open(video_transcript_filename, mode='r')
            video_captions_list = transcript_file.read().split('\n\n')
            # Parse Video Transcript
            for i in range(1, len(video_captions_list)):
                caption_details = video_captions_list[i].split('\n')
                if len(caption_details) > 1 and caption_details[1] != '':
                    if len(video_transcript_parsed) == 0:
                        video_transcript_parsed.append(caption_details[1])
                    elif caption_details[1] != video_transcript_parsed[-1]:
                        video_transcript_parsed.append(caption_details[1])
            # Close the Video Transcript file
            transcript_file.close()
        return video_transcript_parsed

    def preprocess_video_transcript(self, video_captions):
        """
        Method that reads the transcript file of a given video and outputs a list with all the
        video's captions
        :param video_captions: a list of the video captions
        :return:
        """
        # Read Video Transcript file
        video_transcript_processed = list()
        if len(video_captions) > 0:
            for caption in video_captions:
                video_transcript_processed.append(self.preprocess_text(text=caption))
            return ' '.join(video_transcript_processed).replace('\n', '')
        return ''

    def get_video_transcript_features(self):
        """
        Method that gets the Video Transcript features for all all_videos in our Ground Truth
        :return:
        """
        all_video_transcript_features = list()
        if not os.path.isfile(self.VIDEO_TRANSCRIPT_FEATURES_FILENAME):
            progress = tqdm(total=len(self.GROUNDTRUTH_VIDEOS))
            for video_id in self.GROUNDTRUTH_VIDEOS:
                # Read Video Captions
                video_transcript = self.groundtruth_videos_transcripts_col.find_one({'id': video_id})
                # Preprocess Video Captions
                all_video_transcript_features.append(self.preprocess_video_transcript(video_captions=video_transcript['captions']))
                progress.update(1)
            progress.close()

            # Save them to file
            pickle.dump(all_video_transcript_features, open(self.VIDEO_TRANSCRIPT_FEATURES_FILENAME, mode='wb'))
        else:
            # Read saved features from file
            all_video_transcript_features = pickle.load(open(self.VIDEO_TRANSCRIPT_FEATURES_FILENAME, mode='rb'))
        return all_video_transcript_features

    def read_video_comments(self, video_id):
        """
        Method that returns a list with all the downloaded comments files of the given YouTube Video
        :param video_id: the ID of the video to get its comments
        :return: a list with the comments of the given video
        """
        # Initialiaze Variables
        video_comments = list()
        video_comments_filename = '{}/{}/{}.json'.format(self.VIDEO_COMMENTS_BASE_DIR, video_id, video_id)

        # Read file if not empty
        comment_thread_json_string = '{"all_comments":['
        if os.path.isfile(video_comments_filename) and os.stat(video_comments_filename).st_size > 0:
            with open(video_comments_filename, mode='r') as file:
                comments_cntr = 0
                while True:
                    # Read next N comments (lines in file)
                    next_n_comments = list(itertools.islice(file, 50000))
                    # Check if it is the end of file
                    if not next_n_comments:
                        break

                    for comment_line in next_n_comments:
                        if comments_cntr == 0:
                            comment_thread_json_string += comment_line
                        else:
                            comment_thread_json_string += "," + comment_line
                        comments_cntr += 1

                # Convert comments string to json
                comment_thread_json_string += ']}'
                comments_data = json.loads(comment_thread_json_string)

                # Iterate each top level comment threat
                for top_level_comment_threat in comments_data['all_comments']:
                    comment_details = dict(top_level_comment_threat['snippet']['topLevelComment'])
                    if self.key_exists(comment_details, 'snippet', 'textOriginal'):
                        video_comments.append(comment_details['snippet']['textOriginal'])
                    elif self.key_exists(comment_details, 'snippet', 'textDisplay'):
                        video_comments.append(comment_details['snippet']['textDisplay'])
        return video_comments

    def preprocess_video_comments(self, video_comments):
        """
        Method that pre-process all the comments of a given Video. Basically, it removes
        punctuation marks and lowercase all words
        :param video_id:
        :param video_comments:
        :return:
        """
        preprocessed_video_comments = list()
        for comment in video_comments:
            preprocessed_video_comments.append(self.preprocess_text(text=comment))
        return preprocessed_video_comments

    def get_video_comments_features(self):
        """
        Method that gets the Video Comments 'features' for all all_videos in our Ground Truth
        :return:
        """
        all_video_comments_features = list()
        if not os.path.isfile(self.VIDEO_COMMENTS_FEATURES_FILENAME):
            progress = tqdm(total=len(self.GROUNDTRUTH_VIDEOS))
            for video_id in self.GROUNDTRUTH_VIDEOS:
                # Read Video Comments
                video_comments = self.read_video_comments(video_id=video_id)

                # Video Comments
                all_video_comments_features.append(self.preprocess_video_comments(video_comments=video_comments))

                progress.update(1)
            progress.close()

            # Save them to file
            pickle.dump(all_video_comments_features, open(self.VIDEO_COMMENTS_FEATURES_FILENAME, mode='wb'))
        else:
            # Read saved features from file
            all_video_comments_features = pickle.load(open(self.VIDEO_COMMENTS_FEATURES_FILENAME, mode='rb'))

        return all_video_comments_features

    @staticmethod
    def split_train_test_sets_stratified(labels, test_size):
        """
        Method that splits a given X set of input features and labels in a stratified fashion
        preserving the percentage of samples for each class
        :param labels: the dataset labels
        :param test_size: the proportion of the dataset to include in the test split
        :return: indices_train, indices_test
        """
        # Get the indices of the Science all_videos
        indices_science = [i for i, x in enumerate(labels) if x == 0.0]

        # Get the indices of the Pseudoscience all_videos
        indices_pseudoscience = [i for i, x in enumerate(labels) if x == 1.0]

        #
        # Create the TRAIN and the TEST Sets
        #
        # SCIENCE
        total_science_train = int(len(indices_science) * (1 - test_size))
        indices_train = indices_science[0:total_science_train]
        indices_test = indices_science[total_science_train:len(indices_science)]

        # PSEUDOSCIENCE
        total_pseudoscience_train = int(len(indices_pseudoscience) * (1 - test_size))
        indices_train += indices_pseudoscience[0:total_pseudoscience_train]
        indices_test += indices_pseudoscience[total_pseudoscience_train:len(indices_pseudoscience)]

        print('TOTAL VIDEOS: {} | TRAIN: {}, TEST: {}'.format(len(labels), len(indices_train), len(indices_test)))
        return indices_train, indices_test
