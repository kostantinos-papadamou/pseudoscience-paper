#!/usr/bin/python

import sys
from modules.YouTubeSearch import YouTubeSearchAudit
import os
os.chdir('../')


"""
Initialize variables
"""
# Read User Profile
USER_PROFILE = sys.argv[1]

# Read Search Term
# TODO: Make sure that you replace spaces ' ' with '_' when providing the search term to this script
#       E.g., flat earth -> flat_earth
SEARCH_TERM = sys.argv[2].replace('_', ' ').lower()

""" 
Create a YouTube Search Audit object for the given User 
"""
YOUTUBE_SEARCH_AUDIT_MODULE = YouTubeSearchAudit(user_profile=USER_PROFILE, search_term=SEARCH_TERM)

# Start YouTube Search Audit Experiment
YOUTUBE_SEARCH_AUDIT_MODULE.perform_audit()
sys.exit(0)
