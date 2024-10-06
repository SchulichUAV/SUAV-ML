'''
Entry point into the ML pipeline
Every image must pass through this script to be processed prior to detection
This file process an image to be compatable with the models, while optimizing model processing time

Sources: 
    - https://cs231n.github.io/neural-networks-2/#datapre
'''
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Utils')))
from Helper import process_image

if __name__ == "__main__":
    input_folder = ""
    output_folder = ""
    process_image(input_folder, output_folder)