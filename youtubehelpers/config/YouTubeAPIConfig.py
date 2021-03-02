#!/usr/bin/python

class Config(object):
    """
    Static class that includes the configuration of the YouTube Video metadata downloader
    """
    # General Config
    DOWNLOAD_SNIPPET = True
    DOWNLOAD_VIDEO_TAGS = True
    DOWNLOAD_VIDEO_TRANSCRIPT = True
    DOWNLOAD_VIDEO_COMMENTS = True

    # YouTube Data API Config
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    YOUTUBE_DATA_API_KEY = 'YOUR_YOUTUBE_DATA_API_KEY'  # Used by default when an API Key is not provided to the YouTubeVideoDownloader class

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
