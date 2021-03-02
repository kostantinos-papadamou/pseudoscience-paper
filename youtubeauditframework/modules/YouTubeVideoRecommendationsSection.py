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
