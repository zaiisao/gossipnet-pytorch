import os
import json
import glob
import torch

import numpy as np
from tqdm import tqdm

import torch.utils.data as data

class BeatLoader(data.Dataset):
    """
        Loading detections for 'GossipNet' specfic for VRD dataset

        - this loader should mimic what 'nms' gets as input in 'fnet'
        - for now will limit it to just the detections TODO: add image features later
    """
    def __init__(self, dir):
        """
            Initializing data loader
            Input
                dir: directory where detections for individual images are present
        """
        self.dir = dir

        # reading in the images present
        if os.path.basename(self.dir) == "beatles":
            self.files = glob.glob(os.path.join(self.dir, "gt_intervals", "**", "*.txt"), recursive=True)
        else:
            self.files = glob.glob(os.path.join(self.dir, "gt_intervals", "*.txt"))

        self.data = []
        
        self.load()

    def __getitem__(self, index):
        """
            Returning annotations based on a index
        """

        return self.data[index]
    
    def load(self):
        for index in tqdm(range(len(self.files))):
            #selected_audio_file = self.files[index]
            #selected_file_name = selected_audio_file.replace('.wav', '.txt')
            selected_file = self.files[index]
            
            data = {}

            gt_boxes = []
            gt_classes = []

            detections = []
            detection_classes = []

            scores = []

            gt_file = open(selected_file)
            gt_lines = gt_file.readlines()
            gt_file.close()

            for gt_line in gt_lines:
                gt_line = gt_line.replace('\n', '')
                extracted_data_from_gt_line = gt_line.split(' ')
                
                gt_box = [float(item) for item in extracted_data_from_gt_line[1:]]
                if extracted_data_from_gt_line[0] == "downbeat":
                    gt_class = 0
                elif extracted_data_from_gt_line[0] == "beat":
                    gt_class = 1

                gt_boxes.append(gt_box)
                gt_classes.append(gt_class)

            #pred_file = open(os.path.join(self.dir, "pred_intervals", selected_file_name))
            pred_file = open(selected_file.replace('gt_intervals', 'pred_intervals'))
            pred_lines = pred_file.readlines()
            pred_file.close()
            
            for pred_line in pred_lines:
                pred_line = pred_line.replace('\n', '')
                extracted_data_from_pred_line = pred_line.split(' ')

                detection = [float(item) for item in extracted_data_from_pred_line[2:]]
                if extracted_data_from_pred_line[0] == "downbeat":
                    detection_class = 0
                elif extracted_data_from_pred_line[0] == "beat":
                    detection_class = 1

                score = float(extracted_data_from_pred_line[1])
                
                detections.append(detection)
                detection_classes.append(detection_class)
                scores.append(score)
                
            data = {
                'gt_boxes': np.array(gt_boxes),
                'gt_classes': np.array(gt_classes),
                'detections': np.array(detections),
                'detection_classes': np.array(detection_classes),
                'scores': np.array(scores),
            }

            self.data.append(data)
        #END for index in tqdm(range(len(self.files)))
            
    @staticmethod
    def collate(items):
        """
            Will specify how the batch items fetched by the data loader are grouped together
            For our case custom writing it for batch_size =1
        """
        # return batch items as a list
        gt_boxes = [torch.from_numpy(s['gt_boxes']) for s in items]
        gt_classes = [torch.from_numpy(s['gt_classes']) for s in items]

        detections = [torch.from_numpy(s['detections']) for s in items]
        detection_classes = [torch.from_numpy(s['detection_classes']) for s in items]
        scores = [torch.from_numpy(s['scores']) for s in items]

        max_num_gts = max(gt_box.shape[0] for gt_box in gt_boxes)
        max_num_detections = max(detection.shape[0] for detection in detections)
        
        if max_num_gts > 0:
            new_gts = torch.ones((len(gt_boxes), max_num_gts, 5)) * -1
            if max_num_gts > 0:
                for idx, gt_box in enumerate(gt_boxes):
                    gt_class = gt_classes[idx]

                    if gt_box.shape[0] > 0:
                        new_gts[idx, :gt_box.shape[0], :4] = gt_box
                        new_gts[idx, :gt_box.shape[0], 4] = gt_class
        else:
            new_gts = torch.ones((len(gt_boxes), 1, 5)) * -1
        
        if max_num_detections > 0:
            new_detections = torch.ones((len(detections), max_num_detections, 6)) * -1
            if max_num_detections > 0:
                for idx, detection_box in enumerate(detections):
                    detection_class = detection_classes[idx]
                    score = scores[idx]

                    if detection_box.shape[0] > 0:
                        new_detections[idx, :detection_box.shape[0], :4] = detection_box
                        new_detections[idx, :detection_box.shape[0], 4] = detection_class
                        new_detections[idx, :detection_box.shape[0], 5] = score
        else:
            new_detections = torch.ones((len(gt_boxes), 1, 6)) * -1

        return new_detections, new_gts

    def __len__(self):
        """
            Returning total number of files/images whose data is available 
        """
        return len(self.data)