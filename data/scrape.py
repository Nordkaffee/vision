from jmd_imagescraper.core import *
import os

# config
root_dir = os.getcwd()
target_dir = "datasets/CoffeaArabica_4"
query = "Coffea Arabica"
max_results = 1000

# query
duckduckgo_search(root_dir, target_dir, query, max_results=max_results)
