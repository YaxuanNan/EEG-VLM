import os
import re

def search_file(directory, pattern):
    matched_files = []
    regex = re.compile(pattern)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if regex.search(file):
                matched_files.append(file)
    return directory, matched_files
