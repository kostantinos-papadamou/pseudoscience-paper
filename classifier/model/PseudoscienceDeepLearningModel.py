#!/usr/bin/python

from classifier.config.ClassifierConfig import Config

# You can use the following Tensorflow versions:
# 1. CPU: tensorflow < 2.0 (recommended: v1.10.1)
# 2. GPU: tensorflow-gpu < 2.0 (recommended: v1.9.0)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Concatenate, Input, InputLayer
from tensorflow.keras.optimizers import Adam


class PseudoscienceDeepLearningModel(object):
    """
    Class that builds the Pseudoscientific Content Detection Deep Learning Model
    """
    def __init__(self):
        # Build the Model
        self.model = self.build_model()
        return

    def build_model(self):
        """
        Builds a Deep Learning Model that detects Pseudoscientific YouTube videos
        considering the Video Snippet, Video Tags, Transcript, and Comments
        :return: Keras Model
        """
        #
        # Build Deep Learning Model
        #
        """ 1. Build VIDEO SNIPPET Model Branch """
        if Config.INPUT_FEATURES_CONFIG['video_snippet']:
            # Create a Sequential Model
            video_snippet_model_branch = Sequential()
            # Add Input Layer
            video_snippet_model_branch.add(InputLayer(input_shape=(Config.EMBESSING_DIM,), name='video_snippet_input'))

        """ 2. Build VIDEO TAGS Model Branch """
        if Config.INPUT_FEATURES_CONFIG['video_tags']:
            # Create a Sequential Model
            video_tags_model_branch = Sequential()
            # Add Input Layer
            video_tags_model_branch.add(InputLayer(input_shape=(Config.EMBESSING_DIM,), name='video_tags_input'))

        """ 3. Build VIDEO TRANSCRIPT Model Branch """
        if Config.INPUT_FEATURES_CONFIG['video_transcript']:
            # Create a Sequential Model
            video_transcript_model_branch = Sequential()
            # Add Input Layer
            video_transcript_model_branch.add(InputLayer(input_shape=(Config.EMBESSING_DIM,), name='video_transcript_input'))

        """ 4. Build VIDEO COMMENTS Model Branch """
        if Config.INPUT_FEATURES_CONFIG['video_comments']:
            # Create a Sequential Model
            video_comments_model_branch = Sequential()
            # Add Input Layer
            video_comments_model_branch.add(InputLayer(input_shape=(Config.EMBESSING_DIM,), name='video_comments_input'))
            video_comments_model_branch.add(Flatten())

        """ [FUSING NETWORK] Concatenate All Model Branches """
        # Get Models Branches output
        model_branches_output = list()
        if Config.INPUT_FEATURES_CONFIG['video_snippet']:
            model_branches_output.append(video_snippet_model_branch.output)
        if Config.INPUT_FEATURES_CONFIG['video_tags']:
            model_branches_output.append(video_tags_model_branch.output)
        if Config.INPUT_FEATURES_CONFIG['video_transcript']:
            model_branches_output.append(video_transcript_model_branch.output)
        if Config.INPUT_FEATURES_CONFIG['video_comments']:
            model_branches_output.append(video_comments_model_branch.output)

        # Add Concatenation Layer, which concatenates the model branches
        if len(model_branches_output) > 1:
            merged_layers = Concatenate(name='fusing_network')(model_branches_output)
        else:
            merged_layers = model_branches_output[0]

        """ 1st Fully Connected (Dense) Layer """
        # Add Fully Connected Layer
        fully_connected_1 = Dense(units=256, activation='relu', name='fully_connected_1')(merged_layers)
        # Add Dropout Layer
        dropout_layer_1 = Dropout(rate=Config.MODEL_DROPOUT, name='dropout_layer_1')(fully_connected_1)

        """ 2nd Fully Connected (Dense) Layer """
        # Add Fully Connected Layer
        fully_connected_2 = Dense(units=128, activation='relu', name='fully_connected_2')(dropout_layer_1)
        # Add Dropout Layer
        dropout_layer_2 = Dropout(rate=Config.MODEL_DROPOUT, name='dropout_layer_2')(fully_connected_2)

        """ 3rd Fully Connected (Dense) Layer """
        # Add Fully Connected Layer
        fully_connected_3 = Dense(units=64, activation='relu', name='fully_connected_3')(dropout_layer_2)
        # Add Dropout Layer
        dropout_layer_3 = Dropout(rate=Config.MODEL_DROPOUT, name='dropout_layer_3')(fully_connected_3)

        """ 4th Fully Connected (Dense) Layer """
        # Add Fully Connected Layer
        fully_connected_4 = Dense(units=32, activation='relu', name='fully_connected_4')(dropout_layer_3)
        # Add Dropout Layer
        dropout_layer_4 = Dropout(rate=Config.MODEL_DROPOUT, name='dropout_layer_4')(fully_connected_4)

        """ Classification Layer (Softmax) """
        # Add Classification (Softmax) Layer
        classification_layer = Dense(units=Config.NB_CLASSES, activation='softmax', name='classification_layer')(dropout_layer_4)

        #
        # Declare Model Input
        #
        model_input = list()
        if Config.INPUT_FEATURES_CONFIG['video_snippet']:
            model_input.append(video_snippet_model_branch.input)
        if Config.INPUT_FEATURES_CONFIG['video_tags']:
            model_input.append(video_tags_model_branch.input)
        if Config.INPUT_FEATURES_CONFIG['video_transcript']:
            model_input.append(video_transcript_model_branch.output)
        if Config.INPUT_FEATURES_CONFIG['video_comments']:
            model_input.append(video_comments_model_branch.input)

        #
        # Create the Full Model
        #
        full_deep_learning_model = Model(model_input, classification_layer)
        # Set Model Optimizer
        optimizer = Adam(lr=Config.LEARNING_RATE)
        # Compile the Model
        full_deep_learning_model.compile(loss=Config.LOSS_FUNCTION, optimizer=optimizer, metrics=['accuracy'])
        # Summarize the Model
        print(full_deep_learning_model.summary())
        return full_deep_learning_model

    def get_model(self):
        """
        Method that returns the created Deep Learning Model
        :return: Keras model
        """
        return self.model