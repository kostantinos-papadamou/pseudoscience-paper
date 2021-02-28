#!/usr/bin/env bash

# Download Video Transcript using youtube-dl (using an HTTPS Proxy)
output="$(youtube-dl -4 $1 --no-check-certificate --skip-download --write-sub --write-auto-sub --sub-lang en --output $2 --proxy $3 2>&1)"
echo $output

# Check Output and print the appropriate message
if [[ "${output}" == *"HTTP Error 429"* ]]; then
    echo "HTTP_ERROR_429"
elif [[ "${output}" == *"WARNING: video doesn't have subtitles"* ]]; then
    echo "FALSE_NO_SUBITLES"
elif [[ "${output}" == *"ERROR: This video is unavailable"* ]]; then
    echo "FALSE_NO_SUBITLES"
elif [[ "${output}" == *"ERROR"* ]]; then
    echo "FALSE_NO_SUBITLES"
else
    echo "TRUE_DOWNLOADED"
fi