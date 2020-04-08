import os
import json

import numpy as np

import torch.utils.data as data

class detVRDLoader(data.Dataset):
    """
        Loading detections for 'GossipNet' specfic for VRD dataset

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

        # read in the json file - detections and ground truth
        data = json.load(open(os.path.join(self.data_dir, selected_file)))

        # convert from list ot numpy arrays
        data['detections'] = np.array(data['detections'])
        data['gt_boxes'] = np.array(data['gt_boxes'])
        data['scores'] = np.array(data['scores'])

        return data

    @staticmethod
    def collate(items):
        """
            Will specify how the batch items fetched by the data loader are grouped together
            For our case custom writing it for batch_size - 1
        """
        # return batch items as a list
        return items

    def __len__(self):
        """
            Returning total number of files/images whose data is available 
        """
        return len(self.files)