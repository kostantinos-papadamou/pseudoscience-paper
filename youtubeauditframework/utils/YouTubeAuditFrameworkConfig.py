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
    DELETE_WATCH_HISTORY_AFTER_DATE = 'DD-MM-YYYY'  # Specific Date and onwards OR None to delete whole activity
    USER_PROFILES_INFO_FILENAME = 'youtubeauditframework/userprofiles/info/user_profiles_info.json'
    USER_PROFILE_DATA_DIR = 'youtubeauditframework/userprofiles/data'
    USER_PROFILES_WATCH_VIDEOS_BASE_DIR = 'youtubeauditframework/userprofiles/info'
    WATCH_HISTORY_VIDEOS_THRESHOLD = 100

    """ General Experiments Config """
    WEBDRIVER_ELEMENT_DELAY = 20  # wait for a maximum of 20 seconds for HTML elements to load

    """ YouTube Data API Config """
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    YOUTUBE_DATA_API_KEY = 'YOUR_YOUTUBE_DATA_API_KEY'
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
    AUDIT_FRAMEWORK_YOUTUBE_VIDEO_RECS_COL = 'audit_framework_youtube_video_recommendations'

    """ YouTube HOMEPAGE Audit Configuration """
    AUDIT_YOUTUBE_HOMEPAGE_LOGFILE = 'youtubeauditframework/logs/youtube_homepage/{}_AuditYouTubeHomepageLogs.json'
    AUDIT_HOMEPAGE_VIDEOS_THRESHOLD = 30
    AUDIT_HOMEPAGE_TOTAL_REPETITIONS = 50

    """ YouTube SEARCH Audit Configuration """
    AUDIT_YOUTUBE_SEARCH_LOGFILE = 'youtubeauditframework/logs/youtube_search/{}_{}_AuditYouTubeSearchLogs.json'
    AUDIT_SEARCH_RESULTS_THRESHOLD = 20
    AUDIT_YOUTUBE_SEARCH_TOTAL_REPETITIONS = 50

    """ YouTube VIDEO RECOMMENDATIONS SECTION Audit (Random Walks) Configuration """
    AUDIT_VIDEO_RECOMMENDATIONS_SECTION_LOGFILE = 'youtubeauditframework/logs/video_recommendations_section/{}_{}_AuditYouTubeVideoRecommendationsSectionLogs.json'
    AUDIT_RANDOM_WALKS_SEARCH_RESULTS_THRESHOLD = 20
    AUDIT_RANDOM_WALKS_MAX_HOPS = 5
    AUDIT_RANDOM_WALKS_TOTAL_REPETITIONS = 50
    AUDIT_RANDOM_WALKS_WATCH_VIDEO = True
    AUDIT_RANDOM_WALKS_WATCH_VIDEO_PERCENTAGE = 50
    AUDIT_RANDOM_WALKS_RECOMMENDED_VIDEOS_THRESHOLD = 10

    """ Audit Experiments Analysis Configuration """
    AUDIT_ANALYSIS_CONSIDER_UNIQUE_VIDEOS_ONLY = True
    # AUDIT_FRAMEWORK_PLOTS_DIR = ''