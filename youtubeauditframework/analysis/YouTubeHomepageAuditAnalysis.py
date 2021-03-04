#!/usr/bin/python

import sys
from tqdm import tqdm
from pymongo import MongoClient

from youtubeauditframework.utils.Utils import Utils
from youtubeauditframework.utils.YouTubeAuditFrameworkConfig import Config
from youtubeauditframework.analysis.plots.Plots import Plots


class YouTubeHomepageAuditAnalysis(object):
    """
    Class that analyzes the results of the YouTube Homepage Audit experiments.
    This class also implements the required methods to generate plots as in the paper.
    """
    def __init__(self, legend_labels_mapping):
        """ MongoDB Configuration """
        # Host and Port
        self.client = MongoClient('localhost', 27017)
        # DB name
        self.db = self.client[Config.DB_NAME]
        # Collections name
        self.audit_framework_videos_col = self.db[Config.AUDIT_FRAMEWORK_VIDEOS_COL]
        self.audit_framework_youtube_homepage_col = self.db[Config.AUDIT_FRAMEWORK_YOUTUBE_HOMEPAGE_COL]

        """ Initialize Other Variables """
        # Get User Profiles <-> Legend Labels mapping
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

    def analyze_audit_experiments(self):
        """
        Method that analyzes the YouTube Homepage Audit experiments repetitions
        considering only unique videos and analyzes the videos incremental for
        each number of top N videos in the YouTube Homepage of a user
        """
        # Iterate each User Profile and calculate its plot values
        for USER_PROFILE in self.USER_PROFILES:
            print('\n--- Analyzing results for USER PROFILE: {}'.format(USER_PROFILE))

            # Get the experiment details for the current USER PROFILE
            experiment_details = self.audit_framework_youtube_homepage_col.find_one({
                'user_profile_type': USER_PROFILE,
                'total_repetitions': Config.AUDIT_HOMEPAGE_TOTAL_REPETITIONS
            }, {'experiment_details': 1, 'experiment_analysis': 1})

            # Ensure that the results exists and the analysis has not already performed
            if not experiment_details:
                print('\n[{}] YouTube Homepage Audit Experiment results cannot be found in MongoDB'.format(USER_PROFILE))
                return None
            if Utils.key_exists(experiment_details, 'experiment_analysis'):
                print('\n[{}] Incremental analysis for the current Homepage Experiment already performed'.format(USER_PROFILE))
                continue

            # Declare variables
            incremental_homepage_experiment_analysis = list()
            progressBar = tqdm(total=Config.AUDIT_HOMEPAGE_VIDEOS_THRESHOLD)
            for n_top_homepage_videos in range(1, Config.AUDIT_HOMEPAGE_VIDEOS_THRESHOLD + 1):
                # Declare necessary variables for calculation
                pseudoscience_videos_found = list()
                all_videos_seen = list()

                # Iterate Experiment Repetitions
                for experiment_repetition in experiment_details['experiment_details']:
                    # Iterate videos of the current repetition
                    for video_id in experiment_repetition['CRAWLED_VIDEOS'][:n_top_homepage_videos]:
                        # Add to the list of seen videos
                        all_videos_seen.append(video_id)
                        # Get Video Label
                        curr_video_label = self.get_video_label(video_id=video_id)
                        if curr_video_label == 'pseudoscience':
                            pseudoscience_videos_found.append(video_id)

                # Calculate analysis results for the current number of homepage videos
                homepage_experiment_analysis = dict()
                homepage_experiment_analysis['total_videos_seen'] = len(all_videos_seen)
                homepage_experiment_analysis['total_unique_videos_seen'] = len(list(set(all_videos_seen)))
                homepage_experiment_analysis['pseudoscience_videos_found'] = pseudoscience_videos_found
                homepage_experiment_analysis['total_pseudoscience_videos_found'] = len(pseudoscience_videos_found)
                homepage_experiment_analysis['total_unique_pseudoscience_videos_found'] = len(list(set(pseudoscience_videos_found)))
                homepage_experiment_analysis['average_pseudoscience_videos_total'] = (len(pseudoscience_videos_found) / len(all_videos_seen)) * 100
                homepage_experiment_analysis['average_pseudoscience_videos_unique'] = (len(list(set(pseudoscience_videos_found))) / len(list(set(all_videos_seen)))) * 100
                # Add to the list with all the results
                incremental_homepage_experiment_analysis.append(homepage_experiment_analysis)

                progressBar.update(1)
            progressBar.close()

            """ Insert YouTube Homepage Audit Analysis results into MongoDB """
            self.audit_framework_youtube_homepage_col.update_one(
                {'user_profile_type': USER_PROFILE},
                {'$set': {'experiment_analysis': incremental_homepage_experiment_analysis}}
            )
        return

    def plot_results(self, videos_step=1, yaxis_lim_top=100):
        """
        Method that plots the percentage of pseudoscientific videos found in our audit experiments
        of YouTube Homepage as the number of hompage videos increases
        :param videos_step: steps of YouTube Videos in x-axis
        :param yaxis_lim_top: the top limit value of the y-axis
        :return:
        """
        # Initialize Variables
        plot_items = list()  # PLOT ITEMS one for each USER PROFILE
        legend_labels = list()

        """ Calculate Plot Items for each USER PROFILE """
        # Iterate each User Profile and calculate its plot values
        for USER_PROFILE in self.USER_PROFILES:
            # Get the experiment details for the current USER PROFILE
            experiment_details = self.audit_framework_youtube_homepage_col.find_one({
                'user_profile_type': USER_PROFILE,
            }, {'experiment_analysis': 1})

            # Iterate Homepage Top N videos analyses
            curr_user_plot_values = list()
            curr_user_plot_values.append(0.0)
            for topN_videos_cntr in range(0, Config.AUDIT_HOMEPAGE_VIDEOS_THRESHOLD + 1, videos_step):
                if topN_videos_cntr == 0:
                    continue

                if Config.AUDIT_ANALYSIS_CONSIDER_UNIQUE_VIDEOS_ONLY:
                    curr_user_plot_values.append(experiment_details['experiment_analysis'][topN_videos_cntr - 1]['average_pseudoscience_videos_unique'])
                else:
                    curr_user_plot_values.append(experiment_details['experiment_analysis'][topN_videos_cntr - 1]['average_pseudoscience_videos_total'])

            # Add Plot Item and its legend label
            plot_items.append(curr_user_plot_values)
            legend_labels.append(self.PLOT_LEGEND_LABELS[USER_PROFILE])

        """ GENERATE PLOT """
        xaxis_labels = [i for i in range(0, Config.AUDIT_HOMEPAGE_VIDEOS_THRESHOLD + 1, videos_step)]
        x_val_start = xaxis_labels[0]
        x_val_end = xaxis_labels[-1]

        if Config.AUDIT_ANALYSIS_CONSIDER_UNIQUE_VIDEOS_ONLY:
            plot_filename = 'youtube_homepage_plot_unique.pdf'
            y_label = '% unique pseudoscientific videos'
        else:
            plot_filename = 'youtube_homepage_plot.pdf'
            y_label = '% pseudoscientific videos'

        Plots.plot(plot_items=plot_items, plot_filename=plot_filename, ylabel=y_label, xlabel='# top homepage videos',
                   x_val_start=x_val_start, x_val_end=x_val_end, x_val_step=videos_step, ylim_top=yaxis_lim_top, legend_labels=legend_labels)
        return
