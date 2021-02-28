#!/usr/bin/env python

import sys
import json
import requests


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


def write_comments_to_file(filename, jsons):
    """
    Method that writes the provided comments in the given filename
    :param filename: The file to append the given comments
    :param jsons: the comments in JSON format to be written in the file
    :return: None
    """
    with open(filename, mode="a", encoding='utf-8') as f:
        # Check if there are any comments
        try:
            for js in jsons['items']:
                f.write(json.dumps(js) + '\n')
        except KeyError:
            return


def get_video_comments(video_id, API_KEY, nextPageToken):
    """
    A function that takes as parameters the videoid of the YouTube video and your API_KEY
    and request for the specific comments page of the given video
    QUOTA COST: 4 * 2 = 8 maximum per video
    :param video_id:
    :param API_KEY: the YouTube Data API Key
    :param nextPageToken: the
    :return: Video comments page
    """
    HTTP_REQUEST = "https://www.googleapis.com/youtube/v3/commentThreads?pageToken={}&part=snippet%2Creplies&maxResults=100&videoId={}&key={}&order=relevance".format(nextPageToken, video_id, API_KEY)
    r = requests.get(HTTP_REQUEST, stream=True)
    return r


if __name__ == "__main__":
    # The "videoId" option specifies the YouTube video ID that uniquely identifies the video for which the comment will be inserted.
    videoId = sys.argv[1]
    output_dir = sys.argv[2]
    COMMENTS_PAGES_THRESHOLD = sys.argv[3]
    API_KEY = sys.argv[4]

    comment_pages_cntr = 0
    # When results
    nextPageToken = ''
    while True:
        # Download comments with the specific pageToken
        comments_raw = get_video_comments(video_id=videoId, API_KEY=API_KEY, nextPageToken=nextPageToken)
        comments_json = comments_raw.json()

        if 'nextPageToken' not in comments_json:
            nextPageToken = ''
        else:
            nextPageToken = comments_json["nextPageToken"]

        # Create files while you still get http responses
        filename = "%s/%s.json" % (output_dir, videoId)
        write_comments_to_file(filename=filename, jsons=json.loads(comments_raw.text))

        comment_pages_cntr += 1
        if comment_pages_cntr > int(COMMENTS_PAGES_THRESHOLD):
            break
        if nextPageToken == '':
            break