#!/usr/bin/python

import os
import json
import time
import sys
import errno
from shutil import copyfile
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

from youtubeauditframework.utils.YouTubeAuditFrameworkConfig import Config
from youtubeauditframework.utils.Utils import Utils
from youtubehelpers.YouTubeVideoDownloader import YouTubeVideoDownloader


class BuildUserWatchHistory(object):
    """
    Class that creates the Watch History of a given YouTube User Profile by watching a predefined number
    of YouTube Videos (minimum: 100 videos)
    """
    def __init__(self, user_profile):
        # Initialize Variables
        self.USER_PROFILE = user_profile
        self.TIME_TO_SLEEP_BETWEEN_EACH_VIDEO = 5  # seconds

        #
        # Configure Selenium ChromeDriver
        #
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

        # Create a YouTube Video Helper
        self.YOUTUBE_DOWNLOADER = YouTubeVideoDownloader()
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

    def get_user_profile_watch_videos_from_file(self):
        """
        Method that returns a list with all the YouTube Videos IDs that the current
        User Profile will watch to build a watch history
        :return:
        """
        return Utils.read_file(filename='{0}/{1}_watch_history_videos.txt'.format(Config.USER_PROFILES_WATCH_VIDEOS_BASE_DIR, self.USER_PROFILE))

    def get_video_duration(self, video_id):
        """
        Method that returns the duration of the given YouTube Video
        :param video_id: YouTube Video Id
        :return:
        """
        # Get Video Metadata
        video_metadata = self.YOUTUBE_DOWNLOADER.download_video_metadata(video_id=video_id, retrieve_recommended_videos=False)
        # Convert Video duration to seconds
        video_duration_seconds = Utils.convert_youtube_video_duration_to_seconds(video_duration=video_metadata['contentDetails']['duration'])
        return video_duration_seconds

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

    def clear_user_watch_history(self):
        """
        Method that deletes the YouTube's Watch History of the logged-in User Profile
        :return:
        """
        # Load YouTube Activity Control Management page
        self.driver.get('https://myactivity.google.com/activitycontrols/youtube?utm_source=my-activity&hl=en')
        time.sleep(3)

        # Click "Deleted Activity by" button
        # self.wait.until(EC.presence_of_element_located((By.XPATH, '/html/body/div[2]/header/div[4]/div[2]/div/c-wiz/div/div/nav/a[3]'))).click()
        try:
            self.wait.until(EC.presence_of_element_located((By.XPATH, '/html/body/div[2]/header/div[4]/div[2]/div/c-wiz/div/div/nav/a[3]'))).click()
        except TimeoutException:
            # Click the other Delete button
            self.wait.until(EC.presence_of_element_located((By.XPATH, '/html/body/c-wiz/div/div[2]/c-wiz[1]/div/div/div[2]/div[2]/div/button'))).click()

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

    def watch_youtube_video(self, video_id):
        """
        Method that receives a specific YouTube Video ID and watch the full video like a normal YouTube user
        :param video_id:
        :return:
        """
        # Load YouTube Video Page
        self.driver.get('https://www.youtube.com/watch?v={}&autoplay=1'.format(video_id))
        time.sleep(3)

        # Watch the Video
        try:
            self.wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/ytd-app/div/ytd-page-manager/ytd-watch-flexy/div[4]/div[1]/div/div[1]/div/div/div/ytd-player/div/div/div[5]/button'))).click()
        except TimeoutException:
            try:
                self.wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/ytd-app/div/ytd-page-manager/ytd-watch-flexy/div[4]/div[1]/div/div[1]/div/div/div/ytd-player/div/div/div[4]/button'))).click()
            except TimeoutException:
                print('[VIDEO: {0}] ERROR: WATCH VIDEO button not found...'.format(video_id))
                pass

        # Like the Video to increase satisfaction score
        try:
            self.wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="top-level-buttons"]/ytd-toggle-button-renderer[1]/a'))).click()
        except TimeoutException:
            print('[VIDEO: {0}] ERROR: Video LIKE button not found...'.format(video_id))
            pass

        # Get Video duration in seconds and then sleep for that time so that we watch the whole video
        video_duration_seconds = self.get_video_duration(video_id=video_id)
        print('--- [VIDEO: {0}] Sleeping for {1} secs to watch the whole video...'.format(video_id, video_duration_seconds))
        time.sleep(video_duration_seconds)
        return

    def build_watch_history(self, watch_videos_list=None, clear_watch_history=True):
        """
        Method that builds the watch history of the current selected YouTube User Profile
        :param watch_videos_list:
        :param clear_watch_history:
        :return:
        """
        # Find the videos that will be used to build the watch history of the user
        if watch_videos_list is None:
            watch_videos_list = self.get_user_profile_watch_videos_from_file()

        # Check if the correct amount of videos to be watched is available
        if len(watch_videos_list) < Config.WATCH_HISTORY_VIDEOS_THRESHOLD:
            print('--- [{0}] Minimum amount of Videos to be watched: {1}. | TOTAL VIDEOS PROVIDED: {2}'.format(self.USER_PROFILE, Config.WATCH_HISTORY_VIDEOS_THRESHOLD, len(watch_videos_list)))
            sys.exit(errno.ECANCELED)

        # Ensure that User is Authenticated
        if not self.is_user_authenticated():
            sys.exit(errno.EAUTH)

        # Clear User Profile History
        if clear_watch_history:
            self.clear_user_watch_history()

        # Build User Profile Watch History
        print('[{0}] Started building Watch History. TOTAL VIDEOS TO WATCH: {1}'.format(self.USER_PROFILE, len(watch_videos_list)))
        watched_videos_cntr = 1
        for video_id in watch_videos_list:
            print('--- {0}/{1}. Watching Video: {2}'.format(watched_videos_cntr, len(watch_videos_list), video_id))
            self.watch_youtube_video(video_id=video_id)
            time.sleep(self.TIME_TO_SLEEP_BETWEEN_EACH_VIDEO)
            watched_videos_cntr += 1
        print('[{0}] Building Watch History has finished!'.format(self.USER_PROFILE))
        return