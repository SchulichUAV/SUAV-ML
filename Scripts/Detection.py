'''
Final combined detection file to be ran at the competition.
This script integrates all saved models and pre-processing steps,
combining everything into the main detection script.
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Utils')))
from Helper import process_image