#!/usr/bin/python

import sys
from modules.YouTubeVideoRecommendationsSection import YouTubeVideoRecommendationsSectionAudit
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
Create a YouTube Video Recommendations Section Audit object for the given User 
"""
YOUTUBE_VIDEO_RECOMMENDATIONS_AUDIT_MODULE = YouTubeVideoRecommendationsSectionAudit(user_profile=USER_PROFILE, search_term=SEARCH_TERM)

# Start YouTube Search Audit Experiment
YOUTUBE_VIDEO_RECOMMENDATIONS_AUDIT_MODULE.perform_audit()
sys.exit(0)
