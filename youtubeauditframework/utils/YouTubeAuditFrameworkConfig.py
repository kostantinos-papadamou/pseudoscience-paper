#!/usr/bin/python

class Config(object):
    """
    Static class that includes the configuration of the YouTube Recommendation Algorithm Audit Framework
    """
    """ ChromeDriver Config """
    HEADLESS = False
    USE_PROXY = True
    USER_AGENT = 'user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
    CHROMEDRIVER_BASE_DIR = 'youtubeauditframework/utils/webdrivers'

    """ User Profiles Config """
    # TODO: This date will be used to delete the watch history of a User Profile, while auditing YouTube's Recommendation Algorithm.
    #       Due to how YouTube's "Delete Watch History" functionality works you have to build the watch history of a user and set
    #       the NEXT DATE of that date in the variable below. Hence, you will also be able to run the experiments correctly the next
    #       date after the one that you built the User Profiles.
    USER_PROFILES_DELETE_WATCH_HISTORY_DATE = 'DD-MM-YYYY'
    USER_PROFILES_INFO_FILENAME = 'youtubeauditframework/userprofiles/info/user_profiles_info.json'
    USER_PROFILE_DATA_DIR = 'youtubeauditframework/userprofiles/data'
    USER_PROFILES_WATCH_VIDEOS_BASE_DIR = 'youtubeauditframework/userprofiles/info'
    WATCH_HISTORY_VIDEOS_THRESHOLD = 100

    """ General Experiments Config """
    WEBDRIVER_ELEMENT_DELAY = 20  # wait for a maximum of 20 seconds for HTML elements to load

    """ YouTube Data API Config """
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    YOUTUBE_DATA_API_KEY = '<YOUR_YOUTUBE_DATA_API_KEY>'
    # What Video Attributes to download
    DOWNLOAD_SNIPPET = True
    DOWNLOAD_VIDEO_TAGS = True
    DOWNLOAD_COMMENTS = False
    DOWNLOAD_TRANSCRIPT = False
    # Set the Number of Comments
    LIMIT_PAGES_COMMENTS = 1  # 200 Comments per page

    """ MongoDB Database & Collections """
    DB_NAME = 'youtube_recommendation_audit'
    AUDIT_FRAMEWORK_VIDEOS_COL = 'audit_framework_videos'
    AUDIT_FRAMEWORK_YOUTUBE_HOMEPAGE_COL = 'audit_framework_youtube_homepage'
    AUDIT_FRAMEWORK_YOUTUBE_SEARCH_COL = 'audit_framework_youtube_search'
    AUDIT_FRAMEWORK_YOUTUBE_VID_RECS_COL = 'audit_framework_youtube_video_recommendations'

    """ Audit Experiments Data Base Directories """
    # YOUTUBE_HOMEPAGE_AUDIT_DATA_DIR = ''
    # YOUTUBE_SEARCH_AUDIT_DATA_DIR = ''
    # YOUTUBE_VIDEO_RECOMMENDATIONS_AUDIT_DATA_DIR = ''

    """ YouTube Homepage Audit Configuration """
    AUDIT_YOUTUBE_HOMEPAGE_LOGFILE = 'youtubeauditframework/logs/youtube_homepage/{}_AuditYouTubeHomepageLogs.json'
    AUDIT_HOMEPAGE_VIDEOS_THRESHOLD = 30
    AUDIT_HOMEPAGE_TOTAL_REPETITIONS = 50

    """ YouTube Search Audit Configuration """

