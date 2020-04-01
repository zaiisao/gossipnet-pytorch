import os
import json

import torch.utils.data as data

class detLoader(data.Dataset):
    """
        Loading detections for 'GossipNet'

        - this loader should mimic what 'nms' gets as input in 'fnet'
        - for now will limit it to just the detections TODO: add image features later
    """
    def __init__(self, data_dir):
        """
            Initializing data loader
            Input
                data_dir: directory where detections for individual images are present
        """
        self.data_dir = data_dir

        # reading in the images present
        self.files = os.listdir(self.data_dir)

    def __getitem__(self, index):
        """
            Returning annotations based on a index
        """
        selected_file = self.files[index]

        # read in the json file
        detections = json.load(open(os.path.join(self.data_dir, selected_file)))

        return detections

    def __len__(self):
        """
            Returning total number of files/images whose data is available 
        """
        return len(self.files)