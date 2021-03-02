#!/usr/bin/python

import os
import sys
import time
import json
import errno
import random
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


class YouTubeVideoRecommendationsSectionAudit(object):
    """
    Class that provides all the methods to perform audit experiments on YouTube's Video Recommendations section with
    logged-in users, non-logged-in users, and the YouTube Data API while assessing the effects of personalization on
    YouTube's Video recommendations.

    More precisely, during this audit experiment, we perform Live Random Walks on YouTube's Recommendation Graph, thus
    simulating the behavior of users casually browsing YouTube while watching videos according to recommendations.
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
        self.audit_framework_youtube_video_recommendations = self.db[Config.AUDIT_FRAMEWORK_YOUTUBE_VIDEO_RECS_COL]

        # Load YouTube Recommendations Monitor latest statusDetails from the file
        self.YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS = self.load_youtube_random_walks_details()

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

    def load_youtube_random_walks_details(self):
        """
        Method that reads the YouTube Live Random Walks Details (latest status) file
        :return: a JSON dict with the contents of the file
        """
        # Read status details from file if it exists
        if os.path.isfile(Config.AUDIT_VIDEO_RECOMMENDATIONS_SECTION_LOGFILE.format(self.AUDIT_SEARCH_TERM.replace(' ', '-'), self.USER_PROFILE)):
            with open(file=Config.AUDIT_VIDEO_RECOMMENDATIONS_SECTION_LOGFILE.format(self.AUDIT_SEARCH_TERM.replace(' ', '-'), self.USER_PROFILE)) as file:
                return dict(json.load(file))

        # Create a new JSON dict and return
        random_walks_details = {
            'STATUS': 'STOPPED',
            'USER_PROFILE_TYPE': self.USER_PROFILE,
            'SEED_SEARCH_TERM': self.AUDIT_SEARCH_TERM,
            'SEARCH_RESULTS_THRESHOLD': Config.AUDIT_RANDOM_WALKS_SEARCH_RESULTS_THRESHOLD,
            'RANDOM_WALK_MAX_HOPS': Config.AUDIT_RANDOM_WALKS_MAX_HOPS,
            'CURRENT_RANDOM_WALK': 0,
            'YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS': [],
            'RANDOM_WALKS_HISTORY': []
        }
        return random_walks_details

    def save_youtube_random_walks_details(self):
        """
        Method that writes the provided YouTube Live Random Walks Audit experiment details in a file
        :return: None
        """
        print(json.dumps(self.YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS, sort_keys=False, ensure_ascii=False, indent=4),
              file=open(file=Config.AUDIT_VIDEO_RECOMMENDATIONS_SECTION_LOGFILE.format(self.AUDIT_SEARCH_TERM.replace(' ', '-'), self.USER_PROFILE), mode='w'))
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

    def get_video_metadata(self, video_id, retrieve_recommended_videos):
        """
        Method that downloads the metadata of the given YouTube Video using YouTube Data API
        :param video_id: YouTube Video Id
        :return:
        """
        # Get Video Metadata
        video_metadata = self.YOUTUBE_VIDEO_DOWNLOADER.download_video_metadata(video_id=video_id, retrieve_recommended_videos=retrieve_recommended_videos)

        # Add additional information
        video_metadata['retrievedAt'] = str(dt.now())
        video_metadata['statistics'] = dict()
        # video_metadata['relatedVideos'] = dict()
        return video_metadata

    def crawl_youtube_video_using_api(self, video_id):
        """
        Method that downloads the metadata of the given YouTube Video and watches the video
        :param video_id:
        :param hop_number:
        :return:
        """
        # Check if Video already exists in MongoDB
        video_metadata = self.audit_framework_videos_col.find_one({'id': video_id})
        if not video_metadata:
            # Get Video Metadata using YouTube Data API
            video_metadata = self.get_video_metadata(video_id=video_id, retrieve_recommended_videos=True)
            if video_metadata is None:
                return None

            # Add Video Annotation information
            video_metadata['classification'] = dict()
            video_metadata['classification']['classification_category'] = None

            # Insert video to MongoDB
            self.audit_framework_videos_col.insert_one(video_metadata)
        else:
            # Retrieve Related Videos only
            video_metadata['relatedVideos'] = self.YOUTUBE_VIDEO_DOWNLOADER.get_recommended_videos(video_id=video_id)

            # Update Video RelatedVideos in MongoDB
            self.audit_framework_videos_col.update_one(
                {'id': video_id},
                {'$set': {'relatedVideos': video_metadata['relatedVideos']}},
                upsert=True
            )
        return video_metadata

    def crawl_watch_youtube_video(self, video_id, hop_number):
        """
        Method that downloads the metadata of the given YouTube Video and watches the video
        :param video_id:
        :param hop_number:
        :return:
        """
        # Find whether we should watch the video or not
        if Config.AUDIT_RANDOM_WALKS_WATCH_VIDEO and hop_number < 5:
            watch_curr_video = True
        else:
            watch_curr_video = False

        # Load YouTube Video Page
        self.driver.get('https://www.youtube.com/watch?v={}&autoplay=1'.format(video_id))
        time.sleep(3)

        # Check if user is Authenticated before proceeding
        if self.USER_PROFILE != 'NO_PERSONALIZATION' and not self.is_user_authenticated():
            exit(1)

        # Check if LiveStream
        try:
            isLivestream = self.wait.until(EC.presence_of_element_located((By.XPATH, '/html/body/ytd-app/div/ytd-page-manager/ytd-watch-flexy/div[4]/div[1]/div/div[5]/div[2]/ytd-video-primary-info-renderer/div/div/div[1]/div[2]/yt-formatted-string'))).text
            if 'started streaming' in isLivestream.lower():
                print('[VIDEO: {}] is a LIVESTREAM. Skipping and choosing another video...'.format(video_id))
                return None
        except TimeoutException:
            pass

        # Start by Watching the Video
        if watch_curr_video:
            try:
                self.wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/ytd-app/div/ytd-page-manager/ytd-watch-flexy/div[4]/div[1]/div/div[1]/div/div/div/ytd-player/div/div/div[4]/button'))).click()
            except TimeoutException:
                try:
                    self.wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/ytd-app/div/ytd-page-manager/ytd-watch-flexy/div[4]/div[1]/div/div[1]/div/div/div/ytd-player/div/div/div[5]/button'))).click()
                except TimeoutException:
                    pass
        # Keep the time needed to crawl the video details
        video_crawl_started = time.time()
        self.driver.execute_script("window.scrollTo(0, 800)")

        """ DOWNLOAD VIDEO METADATA """
        # Check if Video already exists in MongoDB
        video_exists = False
        video_metadata = self.audit_framework_videos_col.find_one({'id': video_id})
        if not video_metadata:
            # Get Video Metadata using YouTube Data API
            video_metadata = self.get_video_metadata(video_id=video_id, retrieve_recommended_videos=False)
            if video_metadata is None:
                return None

            # Add Video Annotation information
            video_metadata['classification'] = dict()
            video_metadata['classification']['classification_category'] = None
        else:
            # Set Video Exists flag
            video_exists = True

        #
        # GET RELATED VIDEOS (no matter if the video exists or not)
        #
        print('--- [VIDEO: {}] GETTING TOP {} RECOMMENDED VIDEOS...'.format(video_id, Config.AUDIT_RANDOM_WALKS_RECOMMENDED_VIDEOS_THRESHOLD))
        related_videos_list = list()
        related_videos_items = self.driver.find_elements_by_xpath('//*[@id="thumbnail"]')
        included_related_videos = 0
        for related_video_item in related_videos_items:
            try:
                related_video_id = related_video_item.get_attribute('href').split('v=')[1]
                if "&" in related_video_id:
                    related_video_id = related_video_id.split('&')[0]
                related_videos_list.append(related_video_id)
                included_related_videos += 1
            except (AttributeError, IndexError):
                continue

            if Config.AUDIT_RANDOM_WALKS_RECOMMENDED_VIDEOS_THRESHOLD == len(related_videos_list):
                break
        print('--- [VIDEO: {0}] TOP {1} RECOMMENDED VIDEOS: {2}'.format(video_id, Config.AUDIT_RANDOM_WALKS_RECOMMENDED_VIDEOS_THRESHOLD, related_videos_list))
        video_metadata['relatedVideos'] = related_videos_list
        video_metadata['updatedAt'] = str(dt.now())

        # STORE VIDEO INFORMATION IN MongoDB
        if not video_exists:
            # Insert Video Details in MongoDB
            self.audit_framework_videos_col.insert_one(video_metadata)
        else:
            # Update Video Details in MongoDB
            self.audit_framework_videos_col.replace_one({'id': video_id}, video_metadata, upsert=True)

        # WATCH VIDEO
        if watch_curr_video:
            # Calculate Video Crawl Duration
            video_crawl_ended = time.time()
            video_crawl_duration_sec = video_crawl_ended - video_crawl_started

            # Read Video Duration
            video_duration_sec = Utils.convert_youtube_video_duration_to_seconds(video_duration=video_metadata['contentDetails']['duration'])
            # Calculate the final watch time percentage to watch
            final_video_duration_sec = int((video_duration_sec * Config.AUDIT_RANDOM_WALKS_WATCH_VIDEO_PERCENTAGE) / 100)
            final_video_duration_sec = final_video_duration_sec - video_crawl_duration_sec

            print('[{0}] - Sleeping for {1} seconds to watch the full VIDEO: {2}'.format(dt.now().strftime("%d-%m-%Y %H:%M:%S"), final_video_duration_sec, video_id))
            time.sleep(final_video_duration_sec)
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
        self.driver.execute_script("window.scrollTo(0, 1500)")
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
                if len(search_result_videos) == Config.AUDIT_RANDOM_WALKS_SEARCH_RESULTS_THRESHOLD:
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
        Method that performs the YouTube Video Recommendations section Audit Experiment for a specific SEARCH TERM for a specific User Profile.
        More precisely, it performs live Random Walks on YouTube's Recommendation Graph starting from videos returned using a specific SEARCH TERM.
        :return:
        """
        print('--- [{}]-[{}] RANDOM WALKS FOR SEARCH_TERM: {} STARTED'.format(dt.now().strftime("%d-%m-%Y %H:%M:%S"), self.USER_PROFILE, self.AUDIT_SEARCH_TERM))

        # Initialiaze variables
        random_walk_cntr = self.YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS['CURRENT_RANDOM_WALK']
        random_walks_history = self.YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS['RANDOM_WALKS_HISTORY']
        random_walks_details = self.YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS['YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS']

        # Perform All Experiment Repetitions
        while random_walk_cntr < Config.AUDIT_RANDOM_WALKS_TOTAL_REPETITIONS:

            # Init current random walks details
            random_walk_started = time.time()
            curr_random_walk_details = dict()
            curr_random_walk_history = ""

            """ DELETE User WATCH HISTORY from the day after the User Profile training """
            if self.USER_PROFILE != 'NO_PERSONALIZATION' and self.USER_PROFILE != 'YOUTUBE_DATA_API':
                self.clear_user_watch_history()

            """ Get current Random Walk Starting Videos (Search YouTube) """
            # Search YouTube
            if self.USER_PROFILE == 'YOUTUBE_DATA_API':
                starting_videos = self.search_youtube_using_api()
            else:
                starting_videos = self.search_youtube()

            """ Choose Random Walk Starting Video randomly from the search results """
            curr_selected_video_metadata = None
            while curr_selected_video_metadata is None:
                curr_selected_video_id = starting_videos[random.randrange(start=0, stop=len(starting_videos), step=1)]

                # Crawl randomly selected Video Metadata
                print('\n[{}]-[RANDOM WALK: {}/{} | HOP: 0/{}] Getting Metadata of VIDEO: {}'.format(self.USER_PROFILE,
                                                                                                     random_walk_cntr + 1,
                                                                                                     Config.AUDIT_RANDOM_WALKS_TOTAL_REPETITIONS,
                                                                                                     Config.AUDIT_RANDOM_WALKS_MAX_HOPS,
                                                                                                     curr_selected_video_id))
                if self.USER_PROFILE == 'YOUTUBE_DATA_API':
                    curr_selected_video_metadata = self.crawl_youtube_video_using_api(video_id=curr_selected_video_id)
                else:
                    curr_selected_video_metadata = self.crawl_watch_youtube_video(video_id=curr_selected_video_id, hop_number=0)

            # Add HOP 0 selected Video Details
            curr_random_walk_details['hop_0'] = dict()
            curr_random_walk_details['hop_0']['video_id'] = curr_selected_video_id
            curr_random_walk_details['hop_0']['label'] = None
            curr_random_walk_details['hop_0']['relatedVideos'] = curr_selected_video_metadata['relatedVideos']
            # Add Video ID to the current Random Walk history
            curr_random_walk_history += '{}_'.format(curr_selected_video_id)  # TODO: change underscore to another character not used in YouTube VideoIds

            """ Perform Random Walk """
            for hop_num in range(1, Config.AUDIT_RANDOM_WALKS_MAX_HOPS + 1):  # LOOP FROM HOP_1 to HOP_5

                """ [STEP 1] Get the Recommended all_videos of the visited Video of the previous Hop """
                previous_hop_video_recommendations = curr_selected_video_metadata['relatedVideos']
                if len(previous_hop_video_recommendations) == 0:
                    break

                """ [STEP 2] Select randomly the Video to visit in the current Hop from among the recommended all_videos of the previous Hop's Video """
                curr_selected_video_metadata = None
                while curr_selected_video_metadata is None:
                    try:
                        curr_selected_video_id = previous_hop_video_recommendations[random.randrange(start=0, stop=len(previous_hop_video_recommendations), step=1)]
                        # Add Video ID to the current Random Walk history
                        curr_random_walk_history += '{}_'.format(curr_selected_video_id)
                    except ValueError:
                        continue

                    """ [STEP 3] Crawl randomly selected current Hop's Video Details """
                    print('[{}]-[RANDOM WALK: {} | HOP: {}/{}] Getting information of VideoID: {}'.format(self.USER_PROFILE, random_walk_cntr + 1, hop_num, Config.AUDIT_RANDOM_WALKS_MAX_HOPS, curr_selected_video_id))
                    if self.USER_PROFILE == 'YOUTUBE_DATA_API':
                        curr_selected_video_metadata = self.crawl_youtube_video_using_api(video_id=curr_selected_video_id)
                    else:
                        curr_selected_video_metadata = self.crawl_watch_youtube_video(video_id=curr_selected_video_id, hop_number=hop_num)

                """ [STEP 4] Add current HOP selected Video Details """
                curr_random_walk_details['hop_{}'.format(hop_num)] = dict()
                curr_random_walk_details['hop_{}'.format(hop_num)]['video_id'] = curr_selected_video_id
                curr_random_walk_details['hop_{}'.format(hop_num)]['label'] = None
                curr_random_walk_details['hop_{}'.format(hop_num)]['relatedVideos'] = curr_selected_video_metadata['relatedVideos']

            # Ensure that this Random Walk is Unique
            isCurrentRandomWalkUnique = True
            for random_walk in random_walks_history:
                if curr_random_walk_history == random_walk:
                    isCurrentRandomWalkUnique = False
                    break

            # Add current Random Walk details to the list of all the Random Walks details
            if isCurrentRandomWalkUnique:
                # Update the list with all the unique random walks history
                random_walks_history.append(curr_random_walk_history)

                # Update Random Walks details with the current Random Walk
                random_walks_details.append(curr_random_walk_details)

                # Increase Random Walks Counter
                random_walk_cntr += 1

                # Update RANDOM WALKS DETAILS file
                self.YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS['STATUS'] = 'RUNNING'
                self.YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS['CURRENT_RANDOM_WALK'] = random_walk_cntr
                self.YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS['RANDOM_WALKS_HISTORY'] = random_walks_history
                self.YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS['YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS'] = random_walks_details
                self.save_youtube_random_walks_details()

            random_walk_ended = time.time()
            print('--- RANDOM WALK took {:.2f} mins.'.format((random_walk_ended - random_walk_started) / 60))

        print('---[{}] RANDOM WALKS using SEARCH_TERM {} COMPLETED!'.format(self.USER_PROFILE, self.AUDIT_SEARCH_TERM))

        # Update RANDOM WALKS DETAILS file
        self.YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS['STATUS'] = 'COMPLETED'
        self.YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS['CURRENT_RANDOM_WALK'] = Config.AUDIT_RANDOM_WALKS_TOTAL_REPETITIONS
        self.save_youtube_random_walks_details()

        # INSERT Random Walks Details to the Database
        random_walks_details_db = dict()
        random_walks_details_db['user_profile_type'] = self.USER_PROFILE
        random_walks_details_db['seed_search_term_topic'] = self.YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS['SEED_SEARCH_TERM']
        random_walks_details_db['search_results_threshold'] = self.YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS['SEARCH_RESULTS_THRESHOLD']
        random_walks_details_db['random_walks_max_hops'] = self.YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS['RANDOM_WALK_MAX_HOPS']
        random_walks_details_db['random_walks_history'] = self.YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS['RANDOM_WALKS_HISTORY']
        random_walks_details_db['random_walks_details'] = self.YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS['YOUTUBE_RANDOM_WALKS_AUDIT_DETAILS']
        self.audit_framework_youtube_video_recommendations.insert_one(random_walks_details_db)
        return