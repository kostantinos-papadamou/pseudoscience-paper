#!/usr/bin/python

import os
import sys
import time
import json
import errno
from datetime import datetime as dt
from shutil import copyfile
from pymongo import MongoClient

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException

from youtubeauditframework.utils.YouTubeAuditFrameworkConfig import Config
from youtubeauditframework.utils.Utils import Utils
from youtubehelpers.YouTubeVideoDownloader import YouTubeVideoDownloader


class YouTubeHomepageAudit(object):
    """
    Class that provides all the methods to perform audit experiments on YouTube's Homepage with
    logged-in users, non-logged-in users, and the YouTube Data API while assessing the effects
    of personalization on YouTube's Video recommendations
    """
    def __init__(self, user_profile):
        # Initialize Variables
        self.USER_PROFILE = user_profile

        """ Configure Selenium ChromeDriver """
        self.driverOptions = ChromeOptions()
        # Set User-Agent
        self.driverOptions.add_argument(Config.USER_AGENT)
        # Set whether headless or not
        if Config.HEADLESS:
            self.driverOptions.add_argument("--headless")
            self.driverOptions.headless = Config.HEADLESS
        # Set HTTPS Proxy Server
        if Config.USE_PROXY:
            user_proxy = self.get_user_proxy_server()
            if user_proxy == 'HOST:PORT':
                exit('[ERROR] Please set correct HTTPS Proxies in: "youtubeauditframework/userprofiles/info/user_profiles_info.json"')
            self.driverOptions.add_argument('--proxy-server={}'.format(user_proxy))
        # Disable Automation Flags
        self.driverOptions.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.driverOptions.add_experimental_option('useAutomationExtension', False)
        self.driverOptions.add_argument('--disable-web-security')
        self.driverOptions.add_argument('--allow-running-insecure-content')

        # Set User Profile unique Data directory
        self.driverOptions.add_argument("user-data-dir={0}/{1}-data".format(Config.USER_PROFILE_DATA_DIR, self.USER_PROFILE))

        # Find ChromeDriver
        self.webdriver_executable = '{0}/chromedriver_{1}'.format(Config.CHROMEDRIVER_BASE_DIR, self.USER_PROFILE)
        if not os.path.isfile(self.webdriver_executable):
            copyfile(src='{0}/chromedriver'.format(Config.CHROMEDRIVER_BASE_DIR), dst=self.webdriver_executable)

        # Create ChromeDriver
        self.driver = webdriver.Chrome(options=self.driverOptions, executable_path=self.webdriver_executable)
        # Maximize Window
        self.driver.maximize_window()
        self.wait = WebDriverWait(self.driver, Config.WEBDRIVER_ELEMENT_DELAY)

        """ MongoDB Configuration """
        # Host and Port
        self.client = MongoClient('localhost', 27017)
        # DB name
        self.db = self.client[Config.DB_NAME]
        # Collections name
        self.audit_framework_videos_col = self.db[Config.AUDIT_FRAMEWORK_VIDEOS_COL]
        self.audit_framework_youtube_homepage_col = self.db[Config.AUDIT_FRAMEWORK_YOUTUBE_HOMEPAGE_COL]

        # Load YouTube Homepage Audit Experiment latest details from the logfile
        self.HOMEPAGE_AUDIT_DETAILS = self.load_youtube_homepage_audit_details()

        """ YOUTUBE DATA API HELPER """
        # Create a YouTube Video Helper
        self.YOUTUBE_VIDEO_DOWNLOADER = YouTubeVideoDownloader()
        return

    def __del__(self):
        # Close Selenium Browser
        self.close_selenium_browser()
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close Selenium Browser
        self.close_selenium_browser()
        return

    def close_selenium_browser(self):
        # Close Selenium browser
        self.driver.close()
        return

    def get_user_proxy_server(self):
        """
        Method that finds the proxy server of the User Profile
        :return:
        """
        user_profiles_info = Utils.read_json_file(filename=Config.USER_PROFILES_INFO_FILENAME)
        for user_profile in user_profiles_info:
            if user_profile['nickname'] == self.USER_PROFILE:
                return user_profile['proxy']
        print('[{0}] Cannot find the HTTPS Proxy server of this User Profile'.format(self.USER_PROFILE))
        sys.exit(errno.ECANCELED)

    def load_youtube_homepage_audit_details(self):
        """
        Method that reads the YouTube Live Random Walks Details (latest status) file
        :return: a JSON dict with the contents of the file
        """
        # Read status details from file if it exists
        if os.path.isfile(Config.AUDIT_YOUTUBE_HOMEPAGE_LOGFILE.format(self.USER_PROFILE)):
            with open(file=Config.AUDIT_YOUTUBE_HOMEPAGE_LOGFILE.format(self.USER_PROFILE)) as file:
                return dict(json.load(file))

        # Create a new JSON dict and return
        homepage_audit_details = {
            'STATUS': 'STOPPED',
            'USER_PROFILE_TYPE': self.USER_PROFILE,
            'HOMEPAGE_VIDEOS_THRESHOLD': Config.AUDIT_HOMEPAGE_VIDEOS_THRESHOLD,
            'EXPERIMENT_TOTAL_REPETITIONS': Config.AUDIT_HOMEPAGE_TOTAL_REPETITIONS,
            'CURRENT_EXPERIMENT_REPETITION': 0,
            'HOMEPAGE_EXPERIMENT_DETAILS': []
        }
        return homepage_audit_details

    def save_youtube_homepage_experiment_details(self):
        """
        Method that writes the provided YouTube Recommendation Monitor details in a file
        :return:
        """
        print(json.dumps(self.HOMEPAGE_AUDIT_DETAILS, sort_keys=False, ensure_ascii=False, indent=4),
              file=open(file=Config.AUDIT_YOUTUBE_HOMEPAGE_LOGFILE.format(self.USER_PROFILE), mode='w'))
        return

    def is_user_authenticated(self):
        """
        Method that verifies whether the user is authenticated or not
        """
        self.driver.get('https://www.youtube.com')
        time.sleep(3)
        try:
            self.wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="avatar-btn"]')))
            print('--- USER IS AUTHENTICATED')
            return True
        except (TimeoutError, NoSuchElementException):
            print('--- USER IS NOT AUTHENTICATED')
            return False

    def get_video_metadata(self, video_id):
        """
        Method that downloads the metadata of the given YouTube Video using YouTube Data API
        :param video_id: YouTube Video Id
        :return:
        """
        # Get Video Metadata
        video_metadata = self.YOUTUBE_VIDEO_DOWNLOADER.download_video_metadata(video_id=video_id, retrieve_recommended_videos=False)

        # Add additional information
        video_metadata['retrievedAt'] = str(dt.now())
        video_metadata['statistics'] = dict()
        video_metadata['relatedVideos'] = dict()
        return video_metadata

    def crawl_youtube_video(self, video_id):
        """
        Method that scrapes the information of a given YouTube Video Id
        :return: the information of the given video
        """
        # Check if user is Authenticated before proceeding
        if self.USER_PROFILE != 'NO_PERSONALIZATION' and not self.is_user_authenticated():
            exit(1)

        """ DOWNLOAD VIDEO INFORMATION """
        # Check if Video already exists in MongoDB
        video_metadata = self.audit_framework_videos_col.find_one({'id': video_id})
        if not video_metadata:
            # Get Video Metadata using YouTube Data API
            video_metadata = self.get_video_metadata(video_id=video_id)
            if video_metadata is None:
                return None

            # Add Video Annotation information
            video_metadata['classification'] = dict()
            video_metadata['classification']['classification_category'] = None

            # Insert video to MongoDB
            self.audit_framework_videos_col.insert_one(video_metadata)
        return video_metadata

    def get_homepage_top_videos(self):
        """
        Method that loads the YouTube Homepage of a user and returns the Top X Video Ids
        :return:
        """
        print('--- [{}] Getting User\'s YouTube Homepage Videos'.format(self.USER_PROFILE))

        # Load YouTube User Homepage
        self.driver.get('https://www.youtube.com/')
        time.sleep(5)

        # Get the Top X Videos from the User's YouTube Homepage
        self.driver.execute_script("window.scrollTo(0, 1700)")
        time.sleep(7)
        homepage_videos = list()
        homepage_video_items = self.driver.find_elements_by_xpath('//*[@id="thumbnail"]')
        for video_item in homepage_video_items:
            try:
                if 'v=' in video_item.get_attribute('href'):
                    if '&list' in video_item.get_attribute('href'):
                        video_id_temp = video_item.get_attribute('href').split('v=')[1]
                        homepage_videos.append(video_id_temp.split('&')[0])
                    else:
                        homepage_videos.append(video_item.get_attribute('href').split('v=')[1])
                if len(homepage_videos) == Config.AUDIT_HOMEPAGE_VIDEOS_THRESHOLD:
                    break
            except (TypeError, StaleElementReferenceException):
                continue
        return homepage_videos

    def perform_audit(self):
        """
        Method that performs the audit of User's YouTube Homepage
        :return:
        """
        print('--- [{}] YOUTUBE HOMEPAGE AUDIT STARTED'.format(self.USER_PROFILE))

        # Initialize variables
        repetitions_cntr = self.HOMEPAGE_AUDIT_DETAILS['CURRENT_EXPERIMENT_REPETITION']
        audit_experiment_details = self.HOMEPAGE_AUDIT_DETAILS['HOMEPAGE_EXPERIMENT_DETAILS']

        # Perform All Experiment Repetitions
        while repetitions_cntr < Config.AUDIT_HOMEPAGE_TOTAL_REPETITIONS:

            # Init current repetition details
            curr_repetition_details = dict()

            print('\n--- [{}]-[{}/{}] Experiment Repetition STARTED'.format(self.USER_PROFILE, repetitions_cntr+1, Config.AUDIT_HOMEPAGE_TOTAL_REPETITIONS))

            # Perform YouTube's Homepage Audit experiment repetition
            user_homepage_top_videos = list()
            while len(user_homepage_top_videos) < Config.AUDIT_HOMEPAGE_VIDEOS_THRESHOLD:
                user_homepage_top_videos = self.get_homepage_top_videos()
            curr_repetition_details['USER_HOMEPAGE_VIDEOS'] = user_homepage_top_videos
            curr_repetition_details['CRAWLED_VIDEOS'] = list()
            curr_repetition_details['CRAWLED_VIDEOS_DETAILS'] = list()

            # Download the metadata of all the videos in the Homepage of the User Profile
            crawled_videos_counter = 1
            for video_id in user_homepage_top_videos:
                print('--- [{}]-[EXP_ID: {}] Crawling Video {}/{} with ID: {}'.format(self.USER_PROFILE, repetitions_cntr+1, crawled_videos_counter, len(user_homepage_top_videos), video_id))

                # Download Video Metadata
                if video_id not in curr_repetition_details['CRAWLED_VIDEOS']:
                    self.get_video_metadata(video_id=video_id)
                    curr_repetition_details['CRAWLED_VIDEOS'].append(video_id)
                    curr_repetition_details['CRAWLED_VIDEOS_DETAILS'].append({
                        'video_id': video_id,
                        'label': None
                    })
                crawled_videos_counter += 1

            # Update Homepage Audit Experiment Logs
            audit_experiment_details.append(curr_repetition_details)
            # Increase Audit Experiment repetition counter
            repetitions_cntr += 1

            # Update Experiment's Details File
            self.HOMEPAGE_AUDIT_DETAILS['STATUS'] = 'RUNNING'
            self.HOMEPAGE_AUDIT_DETAILS['CURRENT_EXPERIMENT_REPETITION'] = repetitions_cntr
            self.HOMEPAGE_AUDIT_DETAILS['HOMEPAGE_EXPERIMENT_DETAILS'] = audit_experiment_details
            self.save_youtube_homepage_experiment_details()

            # Sleep for 10 minutes between each Experiment repetition to avoid the Carry Over Effect
            print('--- [{}] - [{}] Sleeping for 10 minutes before repetition {}'.format(dt.now().strftime("%d-%m-%Y %H:%M:%S"), self.USER_PROFILE, repetitions_cntr + 1))
            time.sleep(10 * 60)

        print('\n--- [{}] YOUTUBE HOMEPAGE AUDIT EXPERIMENT COMPLETED!'.format(self.USER_PROFILE))

        # Update Homepage Audit Experiment Logs
        self.HOMEPAGE_AUDIT_DETAILS['STATUS'] = 'COMPLETED'
        self.HOMEPAGE_AUDIT_DETAILS['CURRENT_EXPERIMENT_REPETITION'] = repetitions_cntr
        self.save_youtube_homepage_experiment_details()

        # INSERT Audit Experiment Details to the Database
        experiment_details_db = dict()
        experiment_details_db['user_profile_type'] = self.USER_PROFILE
        experiment_details_db['homepage_videos_threshold'] = self.HOMEPAGE_AUDIT_DETAILS['HOMEPAGE_VIDEOS_THRESHOLD']
        experiment_details_db['total_repetitions'] = self.HOMEPAGE_AUDIT_DETAILS['CURRENT_EXPERIMENT_REPETITION']
        experiment_details_db['experiment_details'] = self.HOMEPAGE_AUDIT_DETAILS['HOMEPAGE_EXPERIMENT_DETAILS']
        self.audit_framework_youtube_homepage_col.insert_one(experiment_details_db)
        return
