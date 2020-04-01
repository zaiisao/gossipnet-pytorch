import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
	"""
		Single 'block' architecture
	"""
	def __inti__(self):
		"""
			Initializing 'block' architecture
		"""
		# TODO: move architecture specific dimensions to a central configuration file
		self.shortcut_dim = 128
		self.reduced_dim = 32

		# first fully connected layer - input will be the dimensions for each box
		self.fc1 = nn.Sequential(
						nn.Linear(self.shortcut_dim, self.reduced_dim, bias=True),
						nn.ReLU(inplace=True)
					)

	def forward(self, data):
		"""
			Input
				data: Detections, format?
		"""
		block_fc1_feats = self.fc1(data)
		block_fc1_neighbor_feats = block_fc1_feats
		

class GNet(nn.Module):
	"""
		'GossipNet' architecture
	"""
	def __init__(self, num_classes, num_blocks, class_weights=None):
		"""
			Initializing the gossipnet architecture
			Input:
				num_classes: Number of classes in the dataset
				num_blocks: Number of blocks to be defined in the network
				class_weights: ?
		"""
		# ?
		super(GNet, self).__init__()

	def forward(self, data):
		"""
		
		"""
		# since batch size will always remain as 1
		data = data[0]

		dtBoxes = data['detections']
		gtBoxes = data['gt_boxes']

		# getting box information from detections i.e. (x1, y1, w, h, x2, y2, area)
		dtBoxesData = self.getBoxData(dtBoxes)
		gtBoxesData = self.getBoxData(gtBoxes)

		# computing IoU between detections and ground truth
		dt_gt_iou = self.iou(dtBoxesData, gtBoxesData)
		# computing IoU between detections and detections
		dt_dt_iou = self.iou(dtBoxesData, dtBoxesData)


		# input to the first block must be all zero

	def getBoxData(self, boxes):
		"""
			Getting box information (x1, y1, w, h, x2, y2, area) from (x1, y1, x2, y2)
		"""
		x1 = boxes[:, 0].reshape(-1, 1)
		y1 = boxes[:, 1].reshape(-1, 1)
		x2 = boxes[:, 2].reshape(-1, 1)
		y2 = boxes[:, 3].reshape(-1, 1)
		
		width = x2 - x1
		height = y2 - y1
		assert np.all(width > 0) and np.all(height > 0), "rpn boxes are incorrect - height or width is negative"

		area = np.multiply(width, height)

		return (x1, y1, width, height, x2, y2, area)

	def generatePairwiseFeatures(self):
		"""
			Function to compute pairwise features for detections
		"""
		pass

	def multiClassNMS(self):
		"""
			Function to do multiclass NMS during 'training'
		"""
		pass

	def findNeighbours(self):
		"""
			Function to compute neightbours of every detection
		"""
		pass
