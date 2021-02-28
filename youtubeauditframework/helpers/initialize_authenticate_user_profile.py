#!/usr/bin/python

import os
os.chdir('../../')

import sys
import errno
from shutil import copyfile
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait

from youtubeauditframework.utils.config import Config
from youtubeauditframework.utils.Utils import Utils


class InitializeAuthenticateUserProfile(object):
    """
    Class that initializes the data directory for a specific YouTube User Profile
    and at the same time opens the browser so that we can manually authenticate
    the User Profile on YouTube and be ready for running our audit experiments
    """
    def __init__(self, user_profile):
        # Initialize Variables
        self.USER_PROFILE = user_profile

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
            self.driverOptions.add_argument('--proxy-server={}'.format(self.get_user_proxy_server()))
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
        return

    def get_user_proxy_server(self):
        """
        Method that finds the proxy server of the User Profile
        :return:
        """
        user_profiles_info = Utils.read_json_file(filename=Config.USER_PROFILE_DATA_DIR)
        for user_profile in user_profiles_info:
            if user_profile['nickname'] == self.USER_PROFILE:
                return user_profile['proxy']
        print('[{0}] Cannot find the HTTPS Proxy server of this User Profile'.format(self.USER_PROFILE))
        sys.exit(errno.ECANCELED)

    def load_youtube_auth(self):

        # Open YouTube Authentication webpage
        self.driver.get('https://accounts.google.com/signin/v2/identifier?service=youtube&flowName=GlifWebSignIn&flowEntry=ServiceLogin')
        return


if __name__ == '__main__':

    USER_PROFILE = sys.argv[1]

    # Create Object
    obj = InitializeAuthenticateUserProfile(user_profile=USER_PROFILE)

    # Open YouTube Authentication page
    obj.load_youtube_auth()

    # TODO: At this point you need to authenticate the respective User Profile manually and then you
    #       need to install Chrome Adblock Plus plugin to avoid advertisements when running audit experiments.
