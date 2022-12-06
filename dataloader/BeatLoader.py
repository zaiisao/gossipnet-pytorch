import os
import json

import numpy as np

import torch.utils.data as data

class BeatLoader(data.Dataset):
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
        selected_audio_file = self.files[index]
        selected_file_name = selected_audio_file.replace('.wav', '.txt')

        # # read in the json file - detections and ground truth
        # data = json.load(open(os.path.join(self.data_dir, selected_file)))

        # # convert from list ot numpy arrays
        # data['detections'] = np.array(data['detections'])
        # data['scores'] = np.array(data['scores'])

        # # need to pick cluster centers and labels
        # # config 2.
        # # data['detections'] = np.array(data['cluster_centers'])
        # # data['scores'] = np.array(data['cluster_scores'])

        # # config 3. just add this
        # # data['cluster_labels'] = np.array(data['cluster_labels'])

        # data['gt_boxes'] = np.array(data['gt_boxes'])
        
        # data['file_name'] = selected_file
        
        data = {}

        gt_boxes = []
        detections = []
        scores = []

        gt_file = open(os.path.join(self.data_dir, "..", "gt_intervals", selected_file_name))
        gt_lines = gt_file.readlines()
        gt_file.close()
        
        for gt_line in gt_lines:
            gt_line = gt_line.replace('\n', '')
            extracted_data_from_gt_line = gt_line.split(' ')
            
            gt_box = [float(item) for item in extracted_data_from_gt_line[1:]]

            gt_boxes.append(gt_box)

        pred_file = open(os.path.join(self.data_dir, "..", "pred_intervals", selected_file_name))
        pred_lines = pred_file.readlines()
        pred_file.close()
        
        for pred_line in pred_lines:
            pred_line = pred_line.replace('\n', '')
            extracted_data_from_pred_line = pred_line.split(' ')

            detection = [float(item) for item in extracted_data_from_pred_line[2:]]
            score = float(extracted_data_from_pred_line[1])
            
            detections.append(detection)
            scores.append(score)
            
        data = {
            'gt_boxes': np.array(gt_boxes),
            'detections': np.array(detections),
            'scores': np.array(scores),
        }

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