'''
Training of object classifier.
Saves the trained model to "Models" folder.
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Utils')))
from Helper import process_image