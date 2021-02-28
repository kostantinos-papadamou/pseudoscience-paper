#!/usr/bin/python

class Config(object):
    """
    Static class that includes the configuration of the YouTube Video metadata downloader
    """
    # General
    YOUTUBE_DATA_API_KEY = '<YOUR_YOUTUBE_DATA_API_KEY>'  # Used by default when an API Key is not provided to the YouTubeVideoDownloader class
    DOWNLOAD_VIDEO_TRANSCRIPT = True
    DOWNLOAD_VIDEO_COMMENTS = True

    # Set the Number of Comments
    LIMIT_PAGES_COMMENTS = 1  # 200 Comments per page

    # Recommended Videos
    RETRIEVE_RECOMMENDED_VIDEOS = False
    RECOMMENDED_VIDEOS_THRESHOLD = 10

    # HTTPS Proxies for Video Transcript download
    HTTPS_PROXIES_LIST = [
        'HOST:PORT',
        'HOST:PORT',
        'HOST:PORT',
        'HOST:PORT',
        'HOST:PORT',
    ]
