#!/usr/bin/python

from youtubehelpers.config.YouTubeAPIConfig import Config
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from socket import error as SocketError
import time
import os
import glob
import subprocess


class YouTubeVideoDownloader(object):
    """
    Class that downloads among other information, the following required for classification metadata
    of YouTube videos: 1) Video Snippet; 2) Video Tags; 3) Video Transcript; and 4) Video Comments.
    """
    def __init__(self):
        """ YouTube API Configuration """
        # Ensure that the YouTube Data API Key is correctly set
        if Config.YOUTUBE_DATA_API_KEY == 'YOUR_YOUTUBE_DATA_API_KEY':
            exit('[ERROR] Please set your YouTube Data API Key in youtubehelpers/config/YouTubeAPIConfig.py')

        # Initialize YouTube Data API Client
        self.YOUTUBE_API_SERVICE_NAME = Config.YOUTUBE_API_SERVICE_NAME
        self.YOUTUBE_API_VERSION = Config.YOUTUBE_API_VERSION
        self.YOUTUBE_API_KEY = Config.YOUTUBE_DATA_API_KEY
        self.YOUTUBE_API = build(self.YOUTUBE_API_SERVICE_NAME, self.YOUTUBE_API_VERSION, developerKey=self.YOUTUBE_API_KEY)

        """ HTTPS PROXIES """
        self.HTTPS_PROXY_COUNTER = 0
        self.HTTPS_PROXY_USED = 0
        self.HTTPS_PROXY_REQUESTS = 0
        self.HTTPS_PROXY = Config.HTTPS_PROXIES_LIST[self.HTTPS_PROXY_USED]
        if self.HTTPS_PROXY == 'HOST:PORT':
            exit('[ERROR] Please set correct HTTPS Proxies in youtubehelpers/config/YouTubeAPIConfig.py')

        """ Data Directories """
        self.VIDEO_TRANSCRIPT_BASE_DIR = 'videosdata/transcript'
        self.VIDEO_COMMENTS_BASE_DIR = 'videosdata/comments'
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

    def change_https_proxy(self, force_change=False):
        """
        Method that changes the HTTPS Proxy used
        :return:
        """
        # Change Proxy every 10 requests
        if self.HTTPS_PROXY_REQUESTS % 10 == 0 or force_change:
            self.HTTPS_PROXY_REQUESTS = 0
            self.HTTPS_PROXY_COUNTER += 1
            self.HTTPS_PROXY_USED = self.HTTPS_PROXY_COUNTER % len(Config.HTTPS_PROXIES_LIST)

            # Change the HTTP Proxy used
            self.HTTPS_PROXY = Config.HTTPS_PROXIES_LIST[self.HTTPS_PROXY_USED]
        return

    def get_recommended_videos(self, video_id):
        """
        Method to retrieve the recommended videos (IDs) of a given a video
        :param video_id: the YouTube Video ID to get its related videos
        :return: a list of YouTube Video IDs
        """
        # Declare global and other variables
        related_video_ids = list()
        while True:
            # Try to Send the HTTP Request
            try:
                response = self.YOUTUBE_API.search().list(
                    relatedToVideoId=video_id,
                    type="video",
                    part="id",
                    relevanceLanguage="en",
                    maxResults=Config.RECOMMENDED_VIDEOS_THRESHOLD
                ).execute()

                # Get related video ides in an array
                for i in response['items']:
                    related_video_ids.append(i['id']['videoId'])
                return related_video_ids
            except (HttpError, SocketError) as error:
                print('--- HTTP Error occurred while retrieving related all_videos for VideoID: {}. [ERROR]: {}'.format(video_id, error))
                # Sleep for 30 seconds Change API KEY
                time.sleep(30)

    def search_youtube(self, search_term, max_search_results):
        """
        Method that searches YouTube using the YouTube Data API with a predefined SEARCH TERM and
        returns the Video IDs of the top X videos
        :return:
        """
        # Search YouTube
        search_result_videos = list()
        while True:
            try:
                # Call the search.list method to retrieve results matching the specified search term.
                search_response = self.YOUTUBE_API.search().list(
                    q=search_term,
                    type="video",
                    part="id",
                    maxResults=max_search_results,
                    relevanceLanguage='en'
                ).execute()

                # Merge video ids
                for search_result in search_response.get("items", []):
                    search_result_videos.append(search_result["id"]["videoId"])
                return search_result_videos
            except (HttpError, SocketError) as error:
                print('[ERROR] HTTP Error occurred while searching YouTube for: {}'.format(search_term))
                # Sleep for 30 seconds and change API KEY
                time.sleep(30)

    def download_video_metadata(self, video_id, retrieve_recommended_videos=None):
        """
        Method that queries the YouTube Data API and retrieves the details of a given video.
        :param video_id: the YouTube Video for which we want to get its metadata
        :param retrieve_recommended_videos: whether to force the retrieval of the recommended videos of the given YouTube video
        :return:
        """
        # Send HTTP Request to get Video Info
        while True:
            # Send request to get video's information
            try:
                response = self.YOUTUBE_API.videos().list(
                    # part='id,snippet,contentDetails,statistics',
                    part='id,snippet,contentDetails',
                    id=video_id
                ).execute()

                # Get Video Details
                try:
                    mainVideoInformation = response['items'][0]
                except:
                    return None

                # Retrieve Recommended Videos
                if retrieve_recommended_videos is not None and retrieve_recommended_videos:
                    recommended_video_ids = self.get_recommended_videos(video_id=video_id)
                    mainVideoInformation['relatedVideos'] = recommended_video_ids
                elif Config.RETRIEVE_RECOMMENDED_VIDEOS and retrieve_recommended_videos is None:
                    recommended_video_ids = self.get_recommended_videos(video_id=video_id)
                    mainVideoInformation['relatedVideos'] = recommended_video_ids
                else:
                    mainVideoInformation['relatedVideos'] = list()
                return mainVideoInformation
            except (HttpError, SocketError) as error:
                print('--- HTTP Error occurred while retrieving information for VideoID: {0}. [ERROR]: {1}'.format(video_id, error))

                # Sleep for 30 seconds Change API KEY
                time.sleep(30)

    def video_comments_downloaded(self, video_id):
        """
        Method that checks if the comments of a given video has already been downloaded
        :param video_id:
        :return:
        """
        if os.path.isfile('{}/{}/{}.json'.format(self.VIDEO_COMMENTS_BASE_DIR, video_id, video_id)):
            return True
        return False

    def download_video_comments(self, video_id):
        """
        Method that downloads the comments of a given YouTube Video ID
        :param video_id:
        :return:
        """
        comments_dir = '{}/{}'.format(self.VIDEO_COMMENTS_BASE_DIR, video_id)

        # If Comments Base Directory does not exist download it
        original_umask = os.umask(0)
        try:
            if not os.path.exists(comments_dir):
                os.makedirs(comments_dir, 0o777)
        finally:
            os.umask(original_umask)

        # Call the download_video_comments.py script to get the given video_id's comments
        os.system("python3 youtubescripts/download_video_comments.py {0} {1} {2} {3}".format(video_id, comments_dir, Config.LIMIT_PAGES_COMMENTS, self.YOUTUBE_API_KEY))
        return

    def video_transcript_downloaded(self, video_id):
        """
        Method that check if the video's transcript has already been downloaded
        :param video_id:
        :return:
        """
        video_transcript = glob.glob('{}/{}/{}.*'.format(self.VIDEO_TRANSCRIPT_BASE_DIR, video_id, video_id))
        if len(video_transcript) > 0:
            return True
        return False

    def download_video_transcript(self, video_id):
        """
        Method that downloads the transcript of a given YouTube Video
        :param video_id:
        :return:
        """
        video_url = "https://www.youtube.com/watch?v={}".format(video_id)
        path = "'{0}/{1}/%(id)s.%(ext)s'".format(self.VIDEO_TRANSCRIPT_BASE_DIR, video_id)

        # Create directory where we will store the video's transcript before proceeding
        original_umask = os.umask(0)
        try:
            if not os.path.exists('{}/{}'.format(self.VIDEO_TRANSCRIPT_BASE_DIR, video_id)):
                os.makedirs('{}/{}'.format(self.VIDEO_TRANSCRIPT_BASE_DIR, video_id), 0o777)
        finally:
            os.umask(original_umask)

        # Download Video Transcript
        try:
            output = subprocess.check_output("bash youtubescripts/download_video_transcript.sh {0} {1} {2}".format(video_url, path, self.HTTPS_PROXY), shell=True)
            if "HTTP_ERROR" in str(output):
                return
            # Increase current HTTP Proxy usage
            self.change_https_proxy(force_change=True)
        except subprocess.CalledProcessError as e:
            pass
        return

    def download_video(self, video_id):
        """
        Method that downloads all the information of a given YouTube Video
        :param video_id: a YouTube Video to download its information
        :return:
        """
        """ VIDEO METADATA """
        video_metadata = self.download_video_metadata(video_id=video_id)
        if video_metadata is None:
            print('ERROR: Video Metadata not available for Video: {0}'.format(video_id))
            return None

        """ VIDEO TRANSCRIPT """
        if Config.DOWNLOAD_VIDEO_TRANSCRIPT and not self.video_transcript_downloaded(video_id=video_id):
            self.download_video_transcript(video_id=video_id)

        """ VIDEO COMMENTS """
        if Config.DOWNLOAD_VIDEO_COMMENTS and not self.video_comments_downloaded(video_id=video_id):
            self.download_video_comments(video_id=video_id)

        return video_metadata
