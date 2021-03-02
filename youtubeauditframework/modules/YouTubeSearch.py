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
    def __init__(self, user_profile):
        # Initialize Variables
        self.USER_PROFILE = user_profile
