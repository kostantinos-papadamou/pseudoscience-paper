#!/usr/bin/python

import sys
import statistics
from tqdm import tqdm
from pymongo import MongoClient

from youtubeauditframework.utils.Utils import Utils
from youtubeauditframework.utils.YouTubeAuditFrameworkConfig import Config
from youtubeauditframework.analysis.plots.Plots import Plots


class YouTubeVideoRecommendationsAuditAnalysis(object):
    """
    Class that analyzes the random walks of the YouTube Video Recommendations Audit experiments results.
    This class also implements the required methods to generate plots as in the paper.
    """
    def __init__(self, legend_labels_mapping, search_terms):
        """
        Constructor
        :param legend_labels_mapping: the mapping between the User Profile and its legend label
        :param search_terms: the search terms for which to analyze experiments and generate plots
        """
        """ MongoDB Configuration """
        # Host and Port
        self.client = MongoClient('localhost', 27017)
        # DB name
        self.db = self.client[Config.DB_NAME]
        # Collections name
        self.audit_framework_videos_col = self.db[Config.AUDIT_FRAMEWORK_VIDEOS_COL]
        self.audit_framework_youtube_video_recommendations = self.db[Config.AUDIT_FRAMEWORK_YOUTUBE_VIDEO_RECS_COL]

        """ Initialize Other Variables """
        # Get User Profiles <-> Legend Labels mapping
        self.CONSIDERED_SEARCH_TERMS = search_terms
        self.PLOT_LEGEND_LABELS = legend_labels_mapping
        self.USER_PROFILES = self.get_experiments_users_profiles()
        return

    def get_experiments_users_profiles(self):
        """
        Method that a returns a list of all the User Profiles used to perform the experiments
        :return:
        """
        # user_profiles_info = Utils.read_json_file(filename=Config.USER_PROFILES_INFO_FILENAME)
        # return [user_profile['nickname'] for user_profile in user_profiles_info]
        return [key for key in self.PLOT_LEGEND_LABELS.keys()]

    def get_video_label(self, video_id):
        """
        Method that returns the label of the given YouTube Video from MongoDB
        """
        video_details = self.audit_framework_videos_col.find_one({'id': video_id}, {'classification': 1})
        if Utils.key_exists(video_details, 'classification'):
            return video_details['classification']['classification_category']
        else:
            print('[VIDEO: {}] Video not classified yet. Exiting...'.format(video_id))
            sys.exit(0)

    def analyze_audit_experiments(self, random_walks_starting_hop=0):
        """
        Method that analyzes the random walks of the YouTube Video Recommendations audit experiments
        :param random_walks_starting_hop:
        :return:
        """
        # Iterate each User Profile and calculate its plot values
        for USER_PROFILE in self.USER_PROFILES:
            print('\n--- Analyzing Random Walks for USER PROFILE: {}'.format(USER_PROFILE))

            # Iterate through the keywords for each User Profile
            progressBar = tqdm(total=len(self.CONSIDERED_SEARCH_TERMS))
            for SEARCH_TERM in self.CONSIDERED_SEARCH_TERMS:
                # Initialize Variables
                total_pseudoscience_videos_found = 0

                # Get Random Walk details for the current User Profile and Search Term
                curr_random_walk_details = self.audit_framework_youtube_video_recommendations.find_one({
                    '$and': [
                        {'user_profile_type': USER_PROFILE},
                        {'seed_search_term_topic': SEARCH_TERM}
                    ]
                }, {'random_walks_details': 1, 'random_walks_analysis': 1})

                # Ensure that we have performed Random Walks for the requested YouTube Recommendations Monitor Round ID
                if not curr_random_walk_details:
                    print('--- [{}] Personalized Random Walks for SEARCH TERM {} NOT PERFORMED'.format(USER_PROFILE, SEARCH_TERM))
                    return None
                if Utils.key_exists(curr_random_walk_details, 'random_walks_analysis'):
                    print('--- [{}] Analysis of Personalized Random Walks for SEARCH TERM {} ALREADY PERFORMED'.format(USER_PROFILE, SEARCH_TERM))
                    progressBar.update(1)
                    continue

                hops_pseudoscience_videos_found = [list() for i in range(0, Config.AUDIT_RANDOM_WALKS_MAX_HOPS + 1)]
                hops_all_videos_found = [list() for i in range(0, Config.AUDIT_RANDOM_WALKS_MAX_HOPS + 1)]

                # Iterate Random Walks
                for random_walk in curr_random_walk_details['random_walks_details']:

                    # Iterate each Random Walk and calculate what we want
                    for hop_cntr in range(0, Config.AUDIT_RANDOM_WALKS_MAX_HOPS + 1):
                        hops_all_videos_found[hop_cntr].append(random_walk['hop_{}'.format(hop_cntr)]['video_id'])

                        # Get video label
                        curr_video_label = self.get_video_label(video_id=random_walk['hop_{}'.format(hop_cntr)]['video_id'])
                        if curr_video_label == 'pseudoscience':
                            hops_pseudoscience_videos_found[hop_cntr].append(random_walk['hop_{}'.format(hop_cntr)]['video_id'])
                            total_pseudoscience_videos_found += 1

                """ 
                Calculate the percentage of times our Random Walker has found a PSEUDOSCIENCE video at each Hop
                """
                hops_pseudoscience_videos_found_perc = [0.0 for j in range(0, Config.AUDIT_RANDOM_WALKS_MAX_HOPS + 1)]
                for hop_cntr in range(start=random_walks_starting_hop, stop=Config.AUDIT_RANDOM_WALKS_MAX_HOPS + 1):

                    # HOP 0
                    hop_percentage_pseudoscience = 0.0
                    if hop_cntr == 0:
                        hop_unique_total_videos = len(list(set(hops_all_videos_found[hop_cntr])))
                        hop_unique_total_pseudoscience = len(list(set(hops_pseudoscience_videos_found[hop_cntr])))
                        hop_percentage_pseudoscience = (int(hop_unique_total_pseudoscience) / float(hop_unique_total_videos)) * 100

                    # ALL OTHER HOPS
                    elif hop_cntr >= 1:
                        all_hops_pseudoscience_videos = list()
                        all_hops_videos = list()
                        for i in range(start=random_walks_starting_hop, stop=hop_cntr + 1):
                            all_hops_pseudoscience_videos += hops_pseudoscience_videos_found[i]
                            all_hops_videos += hops_all_videos_found[i]
                        hop_percentage_pseudoscience = (int(len(list(set(all_hops_pseudoscience_videos)))) / float(len(list(set(all_hops_videos))))) * 100

                    # Set the Percentage of Pseudoscience videos found at the current Hop over all unique videos so far in the Walk
                    hops_pseudoscience_videos_found_perc[hop_cntr] = hop_percentage_pseudoscience

                """ Insert YouTube Video Recommendations Audit (Random Walks) Analysis results into MongoDB """
                random_walks_analysis_results = dict()
                random_walks_analysis_results['total_pseudoscience_videos_found'] = total_pseudoscience_videos_found
                random_walks_analysis_results['hops_pseudoscience_videos_found'] = hops_pseudoscience_videos_found
                random_walks_analysis_results['hops_pseudoscience_videos_found_perc'] = hops_pseudoscience_videos_found_perc

                # Update Database Record for the Random Walks of the current USER PROFILE - SEARCH TERM
                self.audit_framework_youtube_video_recommendations.update_one(
                    {'$and': [{'user_profile_type': USER_PROFILE}, {'seed_search_term_topic': SEARCH_TERM}]},
                    {'$set': {'random_walks_analysis': random_walks_analysis_results}}
                )

                progressBar.update(1)
            progressBar.close()
        return

    def plot_results(self, random_walks_starting_hop=0, yaxis_lim_top=100):
        """
        Method that plots the cummulative percentage of finding a pseudoscientific video at each hop for all random walks performed
        with the considered search terms. It actually plots one line in the plot for each User Profile.
        :param random_walks_starting_hop: the starting hop of the random walk to start the plot
        :param yaxis_lim_top: the top limit value of the y-axis
        :return:
        """
        # Initialize Variables
        plot_items_pseudoscience_found_perc = list()  # PLOT ITEMS one for each USER PROFILE
        legend_labels = list()

        """ Calculate Plot Items for each USER PROFILE """
        # Iterate each User Profile and calculate its plot values
        for USER_PROFILE in self.USER_PROFILES:
            curr_user_pseudoscience_found_all_search_terms_perc = [list() for i in range(start=random_walks_starting_hop, stop=Config.AUDIT_RANDOM_WALKS_MAX_HOPS + 1)]

            # Average the results of all the considered Search Terms for each USER PROFILE
            for SEARCH_TERM in self.CONSIDERED_SEARCH_TERMS:
                # Get Random Walks analysis of the current SEARCH TERM for the current USER PROFILE
                curr_random_walk_analysis_results = self.audit_framework_youtube_video_recommendations.find_one(
                    {'user_profile_type': USER_PROFILE, 'seed_search_term_topic': SEARCH_TERM},
                    {'random_walks_analysis': 1})

                # For each iterate all search terms and get the values we want into the arrays with all the values
                for hop_cntr in range(start=random_walks_starting_hop, stop=Config.AUDIT_RANDOM_WALKS_MAX_HOPS + 1):

                    # Append percentage of unique Pseudoscience videos Encountered until each hop
                    curr_user_pseudoscience_found_all_search_terms_perc[hop_cntr-1].append(curr_random_walk_analysis_results['random_walks_analysis']['hops_pseudoscience_videos_found_perc'][hop_cntr])

            # Add the average value for each HOP for all the SEARCH TERMS for the current USER PROFILE into the Plot items lists
            legend_labels.append(self.PLOT_LEGEND_LABELS[USER_PROFILE])
            plot_items_pseudoscience_found_perc.append([statistics.mean(curr_hop_values) for curr_hop_values in curr_user_pseudoscience_found_all_search_terms_perc])

        """ GENERATE PLOT """
        x_val_start = random_walks_starting_hop
        x_val_end = Config.AUDIT_RANDOM_WALKS_MAX_HOPS

        Plots.plot(plot_items=plot_items_pseudoscience_found_perc,
                   plot_filename='{}_random_walks_pseudoscience_found_plot.pdf'.format('_'.join(self.CONSIDERED_SEARCH_TERMS).replace(' ', '_').upper()),
                   ylabel='% unique pseudoscientific videos', xlabel='# hop',
                   x_val_start=x_val_start, x_val_end=x_val_end, x_val_step=1, ylim_top=yaxis_lim_top, legend_labels=legend_labels)
        return
