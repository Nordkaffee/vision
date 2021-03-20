""" Script to scrape images from DuckDuckGo """

import os
import argparse
from jmd_imagescraper.core import *

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target_dir', type=str, default='datasets/scraped/',
                    help='Relative path to where images will be stored.')
parser.add_argument('--query', type=str, default='Coffea Arabica',
                    help='Query to search for in DuckDuckGo.')
parser.add_argument('--max_results', type=int, default=1000,
                    help='Maximum number of images to scrape.')
args = parser.parse_args()

# query images from duck duck go
root_dir = os.getcwd()
duckduckgo_search(
    root_dir,
    args.target_dir,
    args.query,
    max_results=args.max_results
)
