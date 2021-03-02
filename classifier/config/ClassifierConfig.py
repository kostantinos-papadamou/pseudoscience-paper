#!/usr/bin/python

class Config(object):
    """
    Static class that contains the configuration of our Pseudoscientific Content Detection Classifier
    """
    # General Config Variables
    NB_CLASSES = 2  # total number of classes

    # Enable/Disable Model Branches
    INPUT_FEATURES_CONFIG = {
        "video_snippet": True,
        "video_tags": True,
        "video_transcript": True,
        "video_comments": True,
    }

    # Build Deep Learning Model Config
    MODEL_DROPOUT = 0.5
    LEARNING_RATE = 1e-3
    LOSS_FUNCTION = 'binary_crossentropy'  # 'binary_crossentropy' or 'categorical_crossentropy'
    EMBESSING_DIM = 300

    # Train Model Config
    SHUFFLE_TRAIN_SET = True
    OVERSAMPLING = True
    VALIDATION_SPLIT_SIZE = 0.2

    TOTAL_KFOLDS = 10  # number of k-folds (cross-validation)
    NB_EPOCHS = 100
    BATCH_SIZE = 20

    # Model Performance Calculation Config
    CLASSIFICATION_THRESHOLD = 0.7
    PERFORMANCE_SCORES_AVERAGE = 'weighted'
