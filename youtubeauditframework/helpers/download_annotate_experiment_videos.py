#!/usr/bin/python

import os
os.chdir('../../')

from pymongo import MongoClient
from tqdm import tqdm
import time

from pseudoscientificvideosdetection.PseudoscienceClassifier import PseudoscienceClassifier
from youtubehelpers.YouTubeVideoDownloader import YouTubeVideoDownloader


class DownloadAnnotateExperimentsVideos(object):
    """
    Class that downloads all the required information and annotates all the videos encountered during our experiments
    """
    def __init__(self, api_key):
        #
        # MongoDB Configuration
        #
        # Host and Port
        self.client = MongoClient('localhost', 27017)
        # DB name
        self.db = self.client['youtube_recommendation_audit']
        # Collections
        self.audit_framework_videos_col = self.db.audit_framework_videos

        # Create a YouTube Video Downloader Object
        self.VIDEO_DOWNLOADER = YouTubeVideoDownloader(api_key=api_key)

        # Create Video Classifier Object
        self.VIDEO_ANNOTATOR = PseudoscienceClassifier()
        return

    def get_all_notannotated_videos(self):
        """
        Method that returns a list with all the YouTube Video encountered during the
        experiments and have not been annotated
        :return:
        """
        all_notannotated_videos = self.audit_framework_videos_col.find({
            '$and': [
                {'classification.classification_category': None}
            ]
        })
        return [video_info['id'] for video_info in all_notannotated_videos]

    def delete_videos_labels(self):
        """
        Method that deletes the label of all the videos in the collection
        """
        self.audit_framework_videos_col.update_many({}, {'$unset': {'classification': 1}})
        return

    def annotate_videos(self):
        """
        Method that annotates all the non-annotated videos
        :return:
        """
        # Get all not annotated videos
        all_videos = self.get_all_notannotated_videos()

        # Download the information adn annotate videos
        progressBar = tqdm(total=len(all_videos))
        for video_id in all_videos:
            print('\n--- [VIDEO: {}] DOWNLOADING INFORMATION AND ANNOTATING VIDEO'.format(video_id))
            # Get Video Details
            video_details = self.audit_framework_videos_col.find_one({'id': video_id})

            # Download Video Comments
            self.VIDEO_DOWNLOADER.download_video_comments(video_id=video_id)

            # Download Video Transcript
            self.VIDEO_DOWNLOADER.download_video_transcript(video_id=video_id)

            # Annotate Video
            video_label, confidence_score = self.VIDEO_ANNOTATOR.classify(video_details=video_details)

            # Update Video Information
            self.audit_framework_videos_col.update_one({'id': video_id}, {'$set': {'classification.classification_category': video_label}})

            # Sleep to avoid IP address banning when downloading videos' transcript
            # IMPORTANT: Don't change this to less than 5 seconds
            time.sleep(5)
            progressBar.update(1)
        progressBar.close()
        return


if __name__ == '__main__':
    # Init variables
    DELETE_VIDEO_LABELS = False
    ANNOTATE_VIDEOS = True

    # TODO: Set this
    YOUTUBE_DATA_API_KEY = '<YOUR_YOUTUBE_DATA_API_KEY>'
    if YOUTUBE_DATA_API_KEY == '<YOUR_YOUTUBE_DATA_API_KEY>':
        exit('Please fill your YouTube Data API Key')

    # Create a Video Annotator Object
    experimentsVideosAnnotatorObject = DownloadAnnotateExperimentsVideos(api_key=YOUTUBE_DATA_API_KEY)

    # Delete Video Labels
    if DELETE_VIDEO_LABELS:
        experimentsVideosAnnotatorObject.delete_videos_labels()

    # Annotate Experiments Videos
    if ANNOTATE_VIDEOS:
        experimentsVideosAnnotatorObject.annotate_videos()