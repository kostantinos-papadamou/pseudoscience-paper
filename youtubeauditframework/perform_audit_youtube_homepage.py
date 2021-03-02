#!/usr/bin/python

import sys
from modules.YouTubeHomepage import YouTubeHomepageAudit
import os
os.chdir('../')


"""
Initialize variables
"""
# Read User Profile
USER_PROFILE = sys.argv[1]

"""
Create a YouTube Homepage Audit object for the given User
"""
YOUTUBE_HOMEPAGE_AUDIT_MODULE = YouTubeHomepageAudit(user_profile=USER_PROFILE)

# Start YouTube Homepage Audit Experiment
YOUTUBE_HOMEPAGE_AUDIT_MODULE.perform_audit()
sys.exit(0)
