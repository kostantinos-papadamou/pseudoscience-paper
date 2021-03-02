#!/usr/bin/python

from dataset.DatasetUtils import DatasetUtils
from classifier.featureengineering.DataPreparation import DataPreparation
from classifier.model.PseudoscienceDeepLearningModel import PseudoscienceDeepLearningModel
from classifier.config.ClassifierConfig import Config

# Tensorflow < 2.0.0 (installed v1.13.1)
from tensorflow.keras.models import load_model
from keras.callbacks import EarlyStopping

import os
import time
import numpy as np
import pickle
import statistics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix, auc
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical
from shutil import copyfile


class ClassifierTraining(object):
    """
    Class that contains all the necessary code and steps to train our Pseudoscientific Content
    Detection Classifier
    """
    def __init__(self, dataset_object=None, gpu_training=False):
        """
        Constructor
        :param gpu_training: True if train on GPU, False if train on CPU
        """
        # Set CPU/GPU Training
        if gpu_training:
            # Train on GPU
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            # Train on CPU
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Create a Dataset Object
        if dataset_object is not None:
            self.DATASET = dataset_object
        else:
            self.DATASET = DatasetUtils()

        # Create a Data Preparation Object
        self.DATA_PREPARATION = DataPreparation(dataset_object=self.DATASET)

        # Set Base Directories
        self.TRAINING_MODELS_BASE_DIR = 'classifier/training/temp'
        self.BEST_MODEL_BASE_DIR = 'pseudoscientificvideosdetection/models'

        # Create a Pseudoscience Model Object
        self.DEEP_LEARNING_MODEL_OBJ = PseudoscienceDeepLearningModel()

        # Get Dataset Videos and their Labels
        self.dataset_videos, self.videos_classification_details = self.DATASET.get_groundtruth_videos()
        self.dataset_labels = self.DATASET.get_groundtruth_labels()
        self.dataset_labels_categorical = self.DATASET.get_groundtruth_labels_one_hot_encoded(perform_one_hot=False) # [0]
        self.dataset_labels_one_hot = self.DATASET.get_groundtruth_labels_one_hot_encoded(perform_one_hot=True)  # [0., 0., 1., 0.]

        """ Early Stopper (to stop training before overfitting and save best weights) """
        self.early_stopper = EarlyStopping(mode='auto', verbose=2, monitor='val_loss', restore_best_weights=True, patience=20)

        """ Get Ground-truth Video Features """
        print('\n--- Retrieving Model Input Features...')
        # 1. Get VIDEO SNIPPET features (fastText embeddings vector)
        if Config.INPUT_FEATURES_CONFIG['video_snippet']:
            self.video_snippet_features = self.DATA_PREPARATION.get_video_snippet_model_input_features(overwrite=False)

        # 2. Get VIDEO TAGS features (fastText embeddings vector)
        if Config.INPUT_FEATURES_CONFIG['video_tags']:
            self.video_tags_features = self.DATA_PREPARATION.get_video_tags_model_input_features(overwrite=False)

        # 3. Get VIDEO TRANSCRIPT features (fastText embeddings vector)
        if Config.INPUT_FEATURES_CONFIG['video_transcript']:
            self.video_transcript_features = self.DATA_PREPARATION.get_video_transcript_model_input_features(overwrite=False)

        # 4. Get VIDEO COMMENTS features (fastText embeddings vector)
        if Config.INPUT_FEATURES_CONFIG['video_comments']:
            self.video_comments_features = self.DATA_PREPARATION.get_video_comments_model_input_features(overwrite=False)
        return

    @staticmethod
    def calc_roc_auc_curve_metrics(y_true, y_pred_proba):
        """
        Method that calculate the required metrics for the generation  of the ROC AUC Curve
        for a specific model given the predicted probabilities and the labels of the tested videos
        :param y_true: video labels
        :param y_pred_proba: video predictions
        :return:
        """
        preds = y_pred_proba[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true.argmax(axis=1), preds)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds, roc_auc

    def train_model(self):
        """
        Method that trains, stores and tests the Pseudoscience Deep Learning Model
        using K-Fold Cross-Validation. This method will store K trained variances of
        the model in the current directory and in the end will create a copy of the
        best model in 'pseudoscientificvideosdetection/models/' directory
        :return:
        """
        print('/n---Training the Model with {} videos.'.format(len(self.dataset_videos)))

        # Initialize training variables
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        kfold_counter = 1
        folds_predicted_probas, folds_performance_metrics, folds_confusion_matrices = list(), list(), list()
        stratified_kfold = StratifiedKFold(n_splits=Config.TOTAL_KFOLDS, shuffle=True, random_state=None)

        # Train/Test Model with K-Fold Cross Vaildation
        for train_val_set_indices, test_set_indices in stratified_kfold.split(X=self.dataset_videos, y=self.dataset_labels_categorical):
            print('\n--- [K-FOLD %d/%d] TRAIN: %d, TEST: %d' % (kfold_counter, Config.TOTAL_KFOLDS, len(train_val_set_indices), len(test_set_indices)))

            """ TRAIN SET """
            # VIDEO SNIPPET
            X_train_val_snippet = np.take(self.video_snippet_features, indices=train_val_set_indices, axis=0)
            # VIDEO TAGS
            X_train_val_video_tags = np.take(self.video_tags_features, indices=train_val_set_indices, axis=0)
            # VIDEO TRANSCRIPT
            X_train_val_transcript = np.take(self.video_transcript_features, indices=train_val_set_indices, axis=0)
            # VIDEO COMMENTS
            X_train_val_comments = np.take(self.video_comments_features, indices=train_val_set_indices, axis=0)

            # VIDEO LABELS
            Y_train_val_labels = np.take(self.dataset_labels, indices=train_val_set_indices, axis=0)
            Y_train_val_one_hot = np.take(self.dataset_labels_one_hot, indices=train_val_set_indices, axis=0)
            Y_train_val_categorical = np.take(self.dataset_labels_categorical, indices=train_val_set_indices, axis=0)

            """ TEST SET """
            # VIDEO SNIPPET
            X_test_snippet = np.take(self.video_snippet_features, indices=test_set_indices, axis=0)
            # VIDEO TAGS
            X_test_video_tags = np.take(self.video_tags_features, indices=test_set_indices, axis=0)
            # VIDEO TRANSCRIPT
            X_test_transcript = np.take(self.video_transcript_features, indices=test_set_indices, axis=0)
            # VIDEO COMMENTS
            X_test_comments = np.take(self.video_comments_features, indices=test_set_indices, axis=0)

            # VIDEO LABELS
            Y_test_labels = np.take(self.dataset_labels, indices=test_set_indices, axis=0)
            Y_test_one_hot = np.take(self.dataset_labels_one_hot, indices=test_set_indices, axis=0)
            # Y_test_categorical = np.take(self.dataset_labels_categorical, indices=test_set_indices, axis=0)

            """ TRAIN & VALIDATION SETS """
            indices_train, indices_val = self.DATASET.split_train_test_sets_stratified(labels=Y_train_val_categorical, test_size=Config.VALIDATION_SPLIT_SIZE)

            # VIDEO SNIPPET
            X_train_snippet = np.take(X_train_val_snippet, indices=indices_train, axis=0)
            X_val_snippet = np.take(X_train_val_snippet, indices=indices_val, axis=0)
            # VIDEO TAGS
            X_train_video_tags = np.take(X_train_val_video_tags, indices=indices_train, axis=0)
            X_val_video_tags = np.take(X_train_val_video_tags, indices=indices_val, axis=0)
            # VIDEO TRANSCRIPT
            X_train_transcript = np.take(X_train_val_transcript, indices=indices_train, axis=0)
            X_val_transcript = np.take(X_train_val_transcript, indices=indices_val, axis=0)
            # VIDEO COMMENTS
            X_train_comments = np.take(X_train_val_comments, indices=indices_train, axis=0)
            X_val_comments = np.take(X_train_val_comments, indices=indices_val, axis=0)

            # VIDEO LABELS
            Y_val_labels = np.take(Y_train_val_labels, indices=indices_val, axis=0)
            Y_train_one_hot = np.take(Y_train_val_one_hot, indices=indices_train, axis=0)
            Y_val_one_hot = np.take(Y_train_val_one_hot, indices=indices_val, axis=0)
            Y_train_categorical = np.take(Y_train_val_categorical, indices=indices_train, axis=0)
            # Y_val_categorical = np.take(Y_train_val_categorical, indices=indices_val, axis=0)

            """ OVERSAMPLING """
            if Config.OVERSAMPLING:
                print('--- Oversampling Train set...')
                smote = SMOTE(sampling_strategy='not majority')

                # Oversample VIDEO SNIPPET
                X_train_snippet, Y_train_s = smote.fit_resample(X_train_snippet, Y_train_categorical)
                X_train_video_tags, Y_train_s = smote.fit_resample(X_train_video_tags, Y_train_categorical)
                X_train_transcript, Y_train_s = smote.fit_resample(X_train_transcript, Y_train_categorical)
                X_train_comments, Y_train_s = smote.fit_resample(X_train_comments, Y_train_categorical)
                Y_train_oversampled = np.array([to_categorical(label, Config.NB_CLASSES) for label in Y_train_s])
                print('--- [AFTER OVER-SAMPLING] TRAIN: %d, VAL: %d, TEST: %d' % (Y_train_oversampled.shape[0], Y_val_labels.shape[0], Y_test_labels.shape[0]))
            else:
                Y_train_oversampled = Y_train_one_hot

            # """
            # Examine Class Distribution
            # """
            # print('/n--------------------------------------------------------------------')
            # print('Y_TRAIN: %s' % (str(collections.Counter(Y_train_categorical))))
            # print('Y_TRAIN_OVERSAMPLED: %s' % (str(collections.Counter(Y_train_s))))
            # print('Y_VAL: %s' % (str(collections.Counter(Y_val_categorical))))
            # print('Y_TEST: %s' % (str(collections.Counter(Y_test_categorical))))
            # print('--------------------------------------------------------------------\n')

            """ TRAIN THE MODEL """
            # TRAIN SET INPUT
            model_train_input = list()
            if Config.INPUT_FEATURES_CONFIG['video_snippet']:
                model_train_input.append(X_train_snippet)
            if Config.INPUT_FEATURES_CONFIG['video_tags']:
                model_train_input.append(X_train_video_tags)
            if Config.INPUT_FEATURES_CONFIG['video_transcript']:
                model_train_input.append(X_train_transcript)
            if Config.INPUT_FEATURES_CONFIG['video_comments']:
                model_train_input.append(X_train_comments)

            # VALIDATION SET INPUT
            model_val_input = list()
            if Config.INPUT_FEATURES_CONFIG['video_snippet']:
                model_val_input.append(X_val_snippet)
            if Config.INPUT_FEATURES_CONFIG['video_tags']:
                model_val_input.append(X_val_video_tags)
            if Config.INPUT_FEATURES_CONFIG['video_transcript']:
                model_val_input.append(X_val_transcript)
            if Config.INPUT_FEATURES_CONFIG['video_comments']:
                model_val_input.append(X_val_comments)

            # Load Deep Learning Model
            print('\n--- Classifier Training started...')
            model = self.DEEP_LEARNING_MODEL_OBJ.get_model()
            model.fit(model_train_input,
                      Y_train_oversampled,
                      epochs=Config.NB_EPOCHS,
                      batch_size=Config.BATCH_SIZE,
                      validation_data=[model_val_input, Y_val_one_hot],
                      shuffle=Config.SHUFFLE_TRAIN_SET,
                      verbose=1,
                      callbacks=[self.early_stopper])

            """ SAVE TRAINED MODEL """
            print('\n---[K-FOLD {}] Model Training finished. Saving...'.format(kfold_counter))
            final_model_store_path = '{0}/K={1}_pseudoscience_model.hdf5'.format(self.TRAINING_MODELS_BASE_DIR, kfold_counter)
            # Save the whole Model with its weights
            model.save(final_model_store_path)

            """ DELETE and RE-LOAD the MODEL before TESTing """
            del model
            # LOAD THE MODEL
            pseudoscience_model = load_model(final_model_store_path)
            print('--- Model Loaded successfully from directory!')

            """ TEST MODEL """
            model_test_input = list()
            if Config.INPUT_FEATURES_CONFIG['video_snippet']:
                model_test_input.append(X_test_snippet)
            if Config.INPUT_FEATURES_CONFIG['video_tags']:
                model_test_input.append(X_test_video_tags)
            if Config.INPUT_FEATURES_CONFIG['video_transcript']:
                model_test_input.append(X_test_transcript)
            if Config.INPUT_FEATURES_CONFIG['video_comments']:
                model_test_input.append(X_test_comments)

            print('--- Making Predictions on the TEST SET...')
            test_pred_proba = pseudoscience_model.predict(model_test_input, batch_size=Config.BATCH_SIZE, verbose=1, steps=None)
            if Config.CLASSIFICATION_THRESHOLD is not None:
                test_predicted_classes = list()
                for predicted_probas in test_pred_proba:
                    if predicted_probas[1] >= Config.CLASSIFICATION_THRESHOLD:
                        test_predicted_classes.append(1)
                    else:
                        test_predicted_classes.append(0)
            else:
                test_predicted_classes = test_pred_proba.argmax(axis=1)

            """ Calculate Performance Metrics """
            # ACCURACY
            # test_accuracy_without_threshold = accuracy_score(Y_test_one_hot.argmax(axis=1), test_pred_proba.argmax(axis=1))
            test_accuracy = accuracy_score(Y_test_one_hot.argmax(axis=1), test_predicted_classes)
            # PRECISION
            # test_precision_without_threshold = precision_score(Y_test_one_hot.argmax(axis=1), test_pred_proba.argmax(axis=1), average=Config.PERFORMANCE_SCORES_AVERAGE)
            test_precision = precision_score(Y_test_one_hot.argmax(axis=1), test_predicted_classes, average=Config.PERFORMANCE_SCORES_AVERAGE)
            # RECALL
            # test_recall_without_threshold = recall_score(Y_test_one_hot.argmax(axis=1), test_pred_proba.argmax(axis=1), average=Config.PERFORMANCE_SCORES_AVERAGE)
            test_recall = recall_score(Y_test_one_hot.argmax(axis=1), test_predicted_classes, average=Config.PERFORMANCE_SCORES_AVERAGE)
            # F1-SCORE
            # test_f1_score_without_threshold = f1_score(Y_test_one_hot.argmax(axis=1), test_pred_proba.argmax(axis=1), average=Config.PERFORMANCE_SCORES_AVERAGE)
            test_f1_score = f1_score(Y_test_one_hot.argmax(axis=1), test_predicted_classes, average=Config.PERFORMANCE_SCORES_AVERAGE)

            print('\n--- PERFORMANCE METRICS WITH THRESHOLD ---')
            print('--- [KFOLD %d] TEST Accuracy: %.3f' % (kfold_counter, test_accuracy))
            print('--- [KFOLD %d] TEST Precision: %.3f' % (kfold_counter, test_precision))
            print('--- [KFOLD %d] TEST Recall: %.3f' % (kfold_counter, test_recall))
            print('--- [KFOLD %d] TEST F1-Score: %.3f' % (kfold_counter, test_f1_score))

            self.store_kfold_predictions(y_true=Y_test_one_hot, predicted_probas=test_pred_proba, kfold_counter=kfold_counter)
            folds_predicted_probas.append(test_pred_proba)

            # Get ROC Metrics for the current fold
            curr_fpr, curr_tpr, curr_threholds, curr_roc_auc = self.calc_roc_auc_curve_metrics(y_true=Y_test_one_hot, y_pred_proba=test_pred_proba)
            mean_tpr += np.interp(mean_fpr, curr_fpr, curr_tpr)
            mean_tpr[0] = 0.0

            # Append current K-FOLD Performance Metrics to an array that wil later write in file
            folds_performance_metrics.append([test_accuracy, test_precision, test_recall, test_f1_score])

            # Confusion Matrix
            conf_matrix = confusion_matrix(Y_test_one_hot.argmax(axis=1), test_predicted_classes)
            folds_confusion_matrices.append(conf_matrix)

            # Increase Fold Counter
            kfold_counter += 1
            time.sleep(3)

        # Calculate Means for the ROC Curve and store them in a file
        mean_tpr /= Config.TOTAL_KFOLDS
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        pickle.dump([mean_fpr, mean_tpr, mean_auc], file=open('{0}/roc_metrics.p'.format(self.TRAINING_MODELS_BASE_DIR), mode='wb'))

        """
        Calculate mean Accuracy, Precision, Recall, and F1 Score as well as their standard deviation
        """
        print('\n\n--- CROSS VALIDATION MEAN PERFORMANCE METRICS ---')
        # 1. Calculate Mean ACCURACY and its Standard Deviation
        mean_accuracy = statistics.mean([x[0] for x in folds_performance_metrics])
        std_accuracy = statistics.stdev([x[0] for x in folds_performance_metrics])
        print('[ACCURACY] %.3f (%.3f)' % (mean_accuracy, std_accuracy))

        # 2. Calculate Mean PRECISION and its Standard Deviation
        mean_precision = statistics.mean([x[1] for x in folds_performance_metrics])
        std_precision = statistics.stdev([x[1] for x in folds_performance_metrics])
        print('[PRECISION] %.3f (%.3f)' % (mean_precision, std_precision))

        # 3. Calculate Mean RECALL and its Standard Deviation
        mean_recall = statistics.mean([x[2] for x in folds_performance_metrics])
        std_recall = statistics.stdev([x[2] for x in folds_performance_metrics])
        print('[RECALL] %.3f (%.3f)' % (mean_recall, std_recall))

        # 4. Calculate Mean F1-SCORE and its Standard Deviation
        mean_f1score = statistics.mean([x[3] for x in folds_performance_metrics])
        std_f1score = statistics.stdev([x[3] for x in folds_performance_metrics])
        print('[F1-SCORE] %.3f (%.3f)' % (mean_f1score, std_f1score))

        """
        Print the K-FOLD Number that has the best accuracy
        """
        max_acc = 0.0
        best_kfold = -1
        for kfold_number in range(0, len(folds_performance_metrics)):
            if folds_performance_metrics[kfold_number][0] > max_acc:
                max_acc = folds_performance_metrics[kfold_number][0]
                best_kfold = kfold_number + 1  # +1 because we start from 0
        print('\n--- K-FOLD with the BEST PERFORMANCE: {}'.format(best_kfold))
        print('--- BEST FOLD PERFORMANCE METRICS ---\n[ACCURACY] {:.3f}\n[PRECISION] {:.3f}\n[RECALL] {:.3f}\n[F1-SCORE] {:.3f}'.format(
            folds_performance_metrics[best_kfold - 1][0],
            folds_performance_metrics[best_kfold - 1][1],
            folds_performance_metrics[best_kfold - 1][2],
            folds_performance_metrics[best_kfold - 1][3]
        ))
        print('\n--- BEST CONFUSION MATRIX')
        print(folds_confusion_matrices[best_kfold - 1])

        # Copy the best model to the proper folder
        copyfile(src='{0}/K={1}_pseudoscience_model.hdf5'.format(self.TRAINING_MODELS_BASE_DIR, best_kfold),
                 dst='{0}/K={1}_pseudoscience_model.hdf5'.format(self.BEST_MODEL_BASE_DIR, best_kfold))
        return

    def store_kfold_predictions(self, y_true, predicted_probas, kfold_counter):
        """
        Method that stores the predicted probailities of a specific trained model for later use
        :param predicted_probas:
        :param kfold_counter:
        :return:
        """
        # Store current Test Set labels
        test_set_labels_filename = '{0}/K={1}_test_set_labels'.format(self.TRAINING_MODELS_BASE_DIR, kfold_counter)
        pickle.dump(y_true, open(test_set_labels_filename, 'wb'))

        # Store predicted probabilities
        predicted_probabilities_filename = '{0}/K={1}_predicted_probas.p'.format(self.TRAINING_MODELS_BASE_DIR, kfold_counter)
        pickle.dump(predicted_probas, open(predicted_probabilities_filename, 'wb'))
        return
