import json
import os

'''
function takes a results object and appends it's values to a JSON file
if the detection is already in the JSON file, update the lat and long
predictions are all stored in a list of JSON objects each representing
a prediction
'''

'''
now we have to get lat and lon data from
imaging and control systems
'''


def append_lat_lon(dict_obj):

    dict_obj['latitude'] = 'UNAVAILABLE ATM'
    dict_obj['longitude'] = 'UNAVAILABLE ATM'

    return dict_obj
    # get latitude and longitude associate with image


def json_processor(new_data, file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print("creating emty list")
                data = []
    else:
        data = []

    # add longitude and latitute in here
    data.append(append_lat_lon(new_data))

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
