
import sys
from modules.YouTubeHomepage import YouTubeHomepageAudit


# Initialize variables
USER_PROFILE = sys.argv[1]

# Create a YouTube Homepage Audit object for the given User
YOUTUBE_HOMEPAGE_AUDIT_MODULE = YouTubeHomepageAudit(user_profile=USER_PROFILE)

# Start Audit Experiment
YOUTUBE_HOMEPAGE_AUDIT_MODULE.perform_audit()
sys.exit(0)
