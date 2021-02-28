#!/usr/bin/python

class Config(object):
    """
    Static class that includes the configuration of the YouTube Recommendation Algorithm Audit Framework
    """
    # CromeDriver Config
    HEADLESS = False
    USE_PROXY = True
    USER_AGENT = 'user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
    CHROMEDRIVER_BASE_DIR = 'youtubeauditframework/utils/webdrivers'

    # User Profiles Config
    USER_PROFILES_INFO_FILENAME = 'youtubeauditframework/userprofiles/info/user_profiles_info.json'
    USER_PROFILE_DATA_DIR = 'youtubeauditframework/userprofiles/data'
    USER_PROFILES_WATCH_VIDEOS_BASE_DIR = 'youtubeauditframework/userprofiles/info'
    WATCH_HISTORY_VIDEOS_THRESHOLD = 100

    # General Experiments Config
    WEBDRIVER_ELEMENT_DELAY = 20  # wait for a maximum of 20 seconds for HTML elements to load