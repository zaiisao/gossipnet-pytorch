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
	def __init__(self, numClasses, numBlocks, classWeights=None):
		"""
			Initializing the gossipnet architecture
			Input:
				num_classes: Number of classes in the dataset
				num_blocks: Number of blocks to be defined in the network
				class_weights: ?
		"""
		# ?
		super(GNet, self).__init__()

		self.numClasses = numClasses
		self.numBlocks = numBlocks

		self.neighbourIoU = 0.2

	def forward(self, data):
		"""
		
		"""
		# since batch size will always remain as 1
		data = data[0]

		# 15000 boxes are too much for now - reducing - change later
		dtBoxes = data['detections'][:2000]
		gtBoxes = data['gt_boxes']

		# moving computations to tensors for fast computations
		dtBoxes = torch.from_numpy(dtBoxes)
		gtBoxes = torch.from_numpy(gtBoxes)

		# getting box information from detections i.e. (x1, y1, w, h, x2, y2, area)
		dtBoxesData = self.getBoxData(dtBoxes)
		gtBoxesData = self.getBoxData(gtBoxes)

		# computing IoU between detections and ground truth
		dt_gt_iou = self.iou(dtBoxesData, gtBoxesData)
		# computing IoU between detections and detections
		dt_dt_iou = self.iou(dtBoxesData, dtBoxesData)

		# we don't have classes for detections - just the gt_classes
		# so doing single class nms - discuss on this!

		# finding neighbours of all detections
		neighbourPairIds = torch.where(torch.ge(dt_dt_iou, self.neighbourIoU))

		print (neighbourPairIds.shape)

		# input to the first block must be all zero

	@staticmethod
	def intersection(boxes1, boxes2):
		"""
			Compute intersection between all the boxes 
			Output
				Matrix (#boxes1 * #boxes2)
		"""
		boxes1_x1 = boxes1[0]
		boxes1_y1 = boxes1[1]
		boxes1_x2 = boxes1[4]
		boxes1_y2 = boxes1[5]

		boxes2_x1 = boxes2[0].reshape(1, -1)
		boxes2_y1 = boxes2[1].reshape(1, -1)
		boxes2_x2 = boxes2[4].reshape(1, -1)
		boxes2_y2 = boxes2[5].reshape(1, -1)

		x1 = torch.max(boxes1_x1, boxes2_x1)
		y1 = torch.max(boxes1_y1, boxes2_y1)
		x2 = torch.min(boxes1_x2, boxes2_x2)
		y2 = torch.min(boxes1_y2, boxes2_y2)

		width = torch.max(torch.DoubleTensor([0.0]), torch.sub(x2, x1))
		height = torch.max(torch.DoubleTensor([0.0]), torch.sub(y2, y1))

		intersection = torch.mul(width, height)

		return intersection

	@staticmethod
	def iou(boxes1, boxes2):
		"""
			Compute IoU values between boxes1 and boxes2
		"""
		area1 = boxes1[6].reshape(-1, 1) # area1 set in rows
		area2 = boxes2[6].reshape(1, -1) # area2 set in columns

		intersection = 	GNet.intersection(boxes1, boxes2)
		union = torch.sub(torch.add(area1, area2), intersection)
		iou = torch.div(intersection, union)

		return iou

	@staticmethod
	def getBoxData(boxes):
		"""
			Getting box information (x1, y1, w, h, x2, y2, area) from (x1, y1, x2, y2)
		"""
		x1 = boxes[:, 0].reshape(-1, 1)
		y1 = boxes[:, 1].reshape(-1, 1)
		x2 = boxes[:, 2].reshape(-1, 1)
		y2 = boxes[:, 3].reshape(-1, 1)
		
		width = x2 - x1
		height = y2 - y1
		
		# find a torch equivalent or convert to numpy and then check
		# assert np.all(width > 0) and np.all(height > 0), "rpn boxes are incorrect - height or width is negative"

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
