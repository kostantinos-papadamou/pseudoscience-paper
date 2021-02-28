#!/usr/bin/python

import json
import isodate


class Utils(object):
    """
    Utilities class for the YouTube's Recommendation Algorithm Audit Framework
    """
    @staticmethod
    def key_exists(element, *keys):
        """
        Check if *keys (nested) exists in `element` (dict).
        :param keys:
        :return: True if key exists, False if not
        """
        if type(element) is not dict:
            raise AttributeError('keys_exists() expects dict as first argument.')
        if len(keys) == 0:
            raise AttributeError('keys_exists() expects at least two arguments, one given.')

        _element = element
        for key in keys:
            try:
                _element = _element[key]
            except KeyError:
                return False
        return True

    @staticmethod
    def convert_youtube_video_duration_to_seconds(video_duration):
        """
        Method that converts a YouTube Video duration format to seconds
        :param video_duration:
        :return:
        """
        return int(isodate.parse_duration(video_duration).total_seconds())

    @staticmethod
    def read_file(filename):
        """
        Method that reads a TXT file separated by breakline and returns a list with all the rows in the file
        :param filename:
        :return:
        """
        with open(filename) as file:
            file_contents = [line.rstrip('\n') for line in file]
        return file_contents

    @staticmethod
    def read_json_file(filename):
        with open(filename, encoding='utf-8') as file:
            file_contents = list(json.load(file))
        return file_contents