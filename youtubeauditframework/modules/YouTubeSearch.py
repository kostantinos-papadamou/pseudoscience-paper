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


class YouTubeSearchAudit(object):
    """
    Class that provides all the methods to perform audit experiments on YouTube Search Results
    with logged-in users, non-logged-in users, and the YouTube Data API, while also assessing
    the effects of personalization on YouTube's Video recommendations
    """
    def __init__(self, user_profile, search_term):
        """
        Constructor
        :param user_profile: the User Profile nickname to perform the experiment
        :param search_term: the search term to search YouTube
        """
        # Initialize Variables
        self.USER_PROFILE = user_profile
        self.AUDIT_SEARCH_TERM = search_term

        """ Configure Selenium ChromeDriver """
        if self.USER_PROFILE != 'YOUTUBE_DATA_API':
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
        self.audit_framework_youtube_search_col = self.db[Config.AUDIT_FRAMEWORK_YOUTUBE_SEARCH_COL]

        # Load YouTube Homepage Audit Experiment latest details from the logfile
        self.YOUTUBE_SEARCH_AUDIT_DETAILS = self.load_youtube_search_experiment_details()

        """ YOUTUBE DATA API HELPER """
        # Create a YouTube Video Helper
        self.YOUTUBE_VIDEO_DOWNLOADER = YouTubeVideoDownloader()

        return

    def __del__(self):
        # Close Selenium Browser
        if self.USER_PROFILE != 'YOUTUBE_DATA_API':
            self.close_selenium_browser()
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close Selenium Browser
        if self.USER_PROFILE != 'YOUTUBE_DATA_API':
            self.close_selenium_browser()
        return

    def close_selenium_browser(self):
        # Close Selenium browser
        if self.USER_PROFILE != 'YOUTUBE_DATA_API':
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

    def load_youtube_search_experiment_details(self):
        """
        Method that reads the YouTube Live Random Walks Details (latest status) file
        :return: a JSON dict with the contents of the file
        """
        # Read status details from file if it exists
        if os.path.isfile(Config.AUDIT_YOUTUBE_SEARCH_LOGFILE.format(self.AUDIT_SEARCH_TERM.replace(' ', '-'), self.USER_PROFILE)):
            with open(file=Config.AUDIT_YOUTUBE_SEARCH_LOGFILE.format(self.AUDIT_SEARCH_TERM.replace(' ', '-'), self.USER_PROFILE)) as file:
                return dict(json.load(file))

        # Create a new JSON dict and return
        search_experiment_details = {
            'STATUS': 'STOPPED',
            'USER_PROFILE_TYPE': self.USER_PROFILE,
            'SEARCH_TERM': self.AUDIT_SEARCH_TERM,
            'SEARCH_RESULTS_THRESHOLD': Config.AUDIT_SEARCH_RESULTS_THRESHOLD,
            'EXPERIMENT_TOTAL_REPETITIONS': Config.AUDIT_YOUTUBE_SEARCH_TOTAL_REPETITIONS,
            'CURRENT_EXPERIMENT_REPETITION': 0,
            'SEARCH_EXPERIMENT_DETAILS': []
        }
        return search_experiment_details

    def save_youtube_search_experiment_details(self):
        """
        Method that writes the provided YouTube Search Audit experiment details in a file
        :return: None
        """
        print(json.dumps(self.YOUTUBE_SEARCH_AUDIT_DETAILS, sort_keys=False, ensure_ascii=False, indent=4),
              file=open(file=Config.AUDIT_YOUTUBE_SEARCH_LOGFILE.format(self.AUDIT_SEARCH_TERM.replace(' ', '-'), self.USER_PROFILE), mode='w'))
        return

    def clear_user_watch_history(self):
        """
        Method that deletes the YouTube's Watch History of the logged-in User Profile after and onwards the provided date
        :return:
        """
        # Load YouTube Activity Control Management page
        self.driver.get('https://myactivity.google.com/activitycontrols/youtube?utm_source=my-activity&hl=en')
        time.sleep(2)

        # Click "Deleted Activity by" button
        try:
            self.wait.until(EC.presence_of_element_located((By.XPATH, '/html/body/div[2]/header/div[4]/div[2]/div/c-wiz/div/div/nav/a[3]'))).click()
        except TimeoutException:
            # Click the other Delete button
            self.wait.until(EC.presence_of_element_located((By.XPATH, '/html/body/c-wiz/div/div[2]/c-wiz[1]/div/div/div[2]/div[2]/div/button'))).click()
        time.sleep(2)

        if Config.DELETE_WATCH_HISTORY_AFTER_DATE is not None:
            # Select to DELETE CUSTOM RANGE Activity
            self.wait.until(EC.presence_of_element_located((By.XPATH, '/html/body/div[7]/div/div[2]/span/div[2]/div/c-wiz/div/div[3]/ul/li[4]'))).click()
            time.sleep(3)

            # Select After Date from the calendar
            try:
                self.wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[7]/div/div[2]/span/div[2]/div[1]/c-wiz/div/div[3]/div/div[2]/div[1]/div/div[1]/div[1]/div/span/span/div'))).click()
            except TimeoutException:
                self.wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[7]/div/div[2]/span/div[2]/div/c-wiz/div/div[3]/ul/li[4]/div[2]'))).click()
            time.sleep(2)

            try:
                self.driver.find_element_by_xpath("//*[@data-day-of-month='{}']".format(Config.DELETE_WATCH_HISTORY_AFTER_DATE.split('-')[0])).click()
            except TimeoutException:
                self.wait.until(EC.element_to_be_clickable((By.XPATH, "//*[@data-day-of-month='{}']".format(Config.DELETE_WATCH_HISTORY_AFTER_DATE.split('-')[0])))).click()
                pass
            time.sleep(2)

            # Click the DELETE button to proceed
            self.wait.until(EC.presence_of_element_located((By.XPATH, '/html/body/div[7]/div/div[2]/span/div[2]/div[1]/c-wiz/div/div[4]/div/div[2]/button'))).click()
        else:
            # Select to DELETE ALL Activity
            self.wait.until(EC.presence_of_element_located((By.XPATH, '/html/body/div[7]/div/div[2]/span/div[2]/div/c-wiz/div/div[3]/ul/li[3]'))).click()

        # Click the CONFIRM DELETE BUTTON (If it exists because if it does not then it means that there is no activity)
        time.sleep(3)
        try:
            # Click the CONFIRM DELETE BUTTON (If it exists because if it does not then it means that there is no activity)
            self.driver.find_element_by_xpath('/html/body/div[7]/div/div[2]/span/div[2]/div[1]/c-wiz/div/div[4]/div/div[2]/button').click()
        except NoSuchElementException:
            # Let it pass since it means that there is no watch history to delete
            print('[{}] There is no Watch History to delete'.format(self.USER_PROFILE))
            pass
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
        if self.USER_PROFILE != 'NO_PERSONALIZATION' and  self.USER_PROFILE != 'YOUTUBE_DATA_API' and not self.is_user_authenticated():
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

    def search_youtube(self):
        """
        Method that searches YouTube using a predefined SEARCH TERM and returns the Video IDs of the top X videos
        :return:
        """
        print('[{0}] Searching YouTube with SEARCH TERM: {1}'.format(self.USER_PROFILE, self.AUDIT_SEARCH_TERM))

        # Search YouTube using Se
        self.driver.get('https://www.youtube.com/results?search_query={}'.format(self.AUDIT_SEARCH_TERM))
        time.sleep(3)

        # Get the TOP search results
        self.driver.execute_script("window.scrollTo(0, 1300)")
        search_result_videos = list()
        search_result_items = self.driver.find_elements_by_xpath('//*[@id="thumbnail"]')
        for search_result in search_result_items:
            try:
                if 'v=' in search_result.get_attribute('href'):
                    if '&list' in search_result.get_attribute('href'):
                        video_id_temp = search_result.get_attribute('href').split('v=')[1]
                        search_result_videos.append(video_id_temp.split('&')[0])
                    else:
                        search_result_videos.append(search_result.get_attribute('href').split('v=')[1])
                if len(search_result_videos) == Config.AUDIT_SEARCH_RESULTS_THRESHOLD:
                    break
            except (TypeError, StaleElementReferenceException):
                continue
        return search_result_videos

    def search_youtube_using_api(self):
        """
        Method that searches YouTube using the YouTube Data API with a predefined SEARCH TERM and
        returns the Video IDs of the top X videos
        :return:
        """
        print('[{0}] Searching YouTube with SEARCH TERM: {1}'.format(self.USER_PROFILE, self.AUDIT_SEARCH_TERM))

        # Search YouTube
        search_result_videos = self.YOUTUBE_VIDEO_DOWNLOADER.search_youtube(
            search_term=self.AUDIT_SEARCH_TERM,
            max_search_results=self.AUDIT_SEARCH_TERM
        )
        return search_result_videos

    def perform_audit(self):
        """
        Method that performs the YouTube Search Audit Experiment for a specific SEARCH TERM for a specific User Profile
        """
        print('--- [{}] EXPERIMENT FOR SEARCH_TERM: {} STARTED at {}'.format(self.USER_PROFILE, self.AUDIT_SEARCH_TERM, dt.now().strftime("%d-%m-%Y %H:%M:%S")))

        # Initialize variables
        repetitions_cntr = self.YOUTUBE_SEARCH_AUDIT_DETAILS['CURRENT_EXPERIMENT_REPETITION']
        audit_experiment_details = self.YOUTUBE_SEARCH_AUDIT_DETAILS['SEARCH_EXPERIMENT_DETAILS']

        # Clear watch history
        # https://myactivity.google.com/activitycontrols/youtube?utm_source=my-activity&hl=en
        if self.USER_PROFILE != 'NO_PERSONALIZATION' and self.USER_PROFILE != 'YOUTUBE_DATA_API':
            self.clear_user_watch_history()

        # Perform All Experiment Repetitions
        while repetitions_cntr < Config.AUDIT_YOUTUBE_SEARCH_TOTAL_REPETITIONS:

            # Init current repetition details
            curr_repetition_details = dict()

            print('\n--- [{}]-[{}/{}] Experiment repetition for SEARCH TERM {} STARTED'.format(self.USER_PROFILE, repetitions_cntr+1, Config.AUDIT_YOUTUBE_SEARCH_TOTAL_REPETITIONS, self.AUDIT_SEARCH_TERM))

            # Perform YouTube Search Audit experiment repetition
            search_results = list()
            while len(search_results) < Config.AUDIT_SEARCH_RESULTS_THRESHOLD:
                if self.USER_PROFILE == 'YOUTUBE_DATA_API':
                    search_results = self.search_youtube_using_api()
                else:
                    search_results = self.search_youtube()
            curr_repetition_details['SEARCH_RESULTS'] = search_results
            curr_repetition_details['CRAWLED_VIDEOS'] = list()
            curr_repetition_details['CRAWLED_VIDEOS_DETAILS'] = list()

            # Crawl current Experiment Repetition Video Details
            crawled_videos_counter = 1
            for video_id in search_results:

                # Get Video Metadata
                print('--- [{}] SEARCH TERM: {} | [EXP_ID: {}] | Crawling video information {}/{} with ID: {}'.format(self.USER_PROFILE, self.AUDIT_SEARCH_TERM, repetitions_cntr+1, crawled_videos_counter, len(search_results), video_id))
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
            self.YOUTUBE_SEARCH_AUDIT_DETAILS['STATUS'] = 'RUNNING'
            self.YOUTUBE_SEARCH_AUDIT_DETAILS['CURRENT_EXPERIMENT_REPETITION'] = repetitions_cntr
            self.YOUTUBE_SEARCH_AUDIT_DETAILS['SEARCH_EXPERIMENT_DETAILS'] = audit_experiment_details
            self.save_youtube_search_experiment_details()

            # Sleep for 10 minutes between each Experiment repetition to avoid the Carry Over Effect
            print('--- [{}] - [{}] Sleeping for 10 minutes before repetition {}'.format(dt.now().strftime("%d-%m-%Y %H:%M:%S"), self.USER_PROFILE, repetitions_cntr + 1))
            time.sleep(10 * 60)

        print('\n--- [{}] YOUTUBE SEARCH EXPERIMENT using SEARCH_TERM {} COMPLETED!'.format(self.USER_PROFILE, self.AUDIT_SEARCH_TERM))

        # Update YouTube Search Audit Experiment Logs
        self.YOUTUBE_SEARCH_AUDIT_DETAILS['STATUS'] = 'COMPLETED'
        self.YOUTUBE_SEARCH_AUDIT_DETAILS['CURRENT_EXPERIMENT_REPETITION'] = repetitions_cntr
        self.save_youtube_search_experiment_details()

        # INSERT Audit Experiment Details to the Database
        experiment_details_db = dict()
        experiment_details_db['user_profile_type'] = self.USER_PROFILE
        experiment_details_db['search_term'] = self.AUDIT_SEARCH_TERM
        experiment_details_db['search_results_threshold'] = Config.AUDIT_SEARCH_RESULTS_THRESHOLD
        experiment_details_db['total_repetitions'] = self.YOUTUBE_SEARCH_AUDIT_DETAILS['CURRENT_EXPERIMENT_REPETITION']
        experiment_details_db['experiment_details'] = self.YOUTUBE_SEARCH_AUDIT_DETAILS['SEARCH_EXPERIMENT_DETAILS']
        self.audit_framework_youtube_search_col.insert_one(experiment_details_db)
        return

