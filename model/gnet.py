import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
	"""
		Single 'block' architecture
	"""
	def __init__(self):
		"""
			Initializing 'block' architecture
		"""
		super(Block, self).__init__()

		# TODO: move architecture specific dimensions to a central configuration file
		self.shortcutDim = 128
		self.reducedDim = 32

		# FC layer for input detection - input will be the dimensions for each box
		self.fc1 = nn.Sequential(
						nn.Linear(self.shortcutDim, self.reducedDim, bias=True),
						nn.ReLU(inplace=True)
					)

		# FC layers for pairwise features
		self.num_block_pwfeat_fc = 2 # keep it >=1
		self.blockInnerDim = 2 * 32 # why is it defined like this?
		self.block_pwfeat_fc_layers = []
		self.block_pwfeat_fc_layers.append(nn.Sequential(
											nn.Linear(3*32, self.blockInnerDim, bias=True),
											nn.ReLU(inplace=True)
										))
		for _ in range(self.num_block_pwfeat_fc-1):
			self.block_pwfeat_fc_layers.append(nn.Sequential(
												nn.Linear(self.blockInnerDim, self.blockInnerDim, bias=True),
												nn.ReLU(inplace=True)
											))
		self.block_pwfeat_fc_layers = nn.ModuleList(self.block_pwfeat_fc_layers)

		# FC layers for post max-pooling
		self.num_block_pwfeat_postmax_fc = 2
		self.block_pwfeat_postmax_fc_layers = []
		for _ in range(self.num_block_pwfeat_postmax_fc):
			self.block_pwfeat_postmax_fc_layers.append(nn.Sequential(
														nn.Linear(self.blockInnerDim, self.blockInnerDim, bias=True),
														nn.ReLU(inplace=True)
													))
		self.block_pwfeat_postmax_fc_layers = nn.ModuleList(self.block_pwfeat_postmax_fc_layers)

		# making dimensions back to 'shortcutDim'
		self.outputFC = nn.Sequential(
							nn.Linear(self.blockInnerDim, self.shortcutDim, bias=True)
						)
		self.outputReLU = nn.ReLU(inplace=True)

	def forward(self, detFeatures, cIdxs, nIdxs, pairFeatures):
		"""
			Input
				detFeatures: RPN detections features, (#detections, 128)
		"""
		block_fc1_feats = self.fc1(detFeatures)
		block_fc1_neighbor_feats = block_fc1_feats
		
		cFeats = block_fc1_feats[cIdxs]
		nFeats = block_fc1_neighbor_feats[nIdxs]

		combinedFeatures = torch.cat((pairFeatures, cFeats, nFeats), 1)

		for layer in self.block_pwfeat_fc_layers:
			combinedFeatures = layer(combinedFeatures)

		maxPooledFeatures = None

		# max pooling across neighbours (based on cIdxs)
		# don't have a pytorch alternative for segment_max - using a loop for now - TODO: find a better alternative
		numDets = torch.max(cIdxs) + 1 # temp jugaad!
		detCheck = 0
		for i in range(numDets):
			# extracting neighbour features
			detNeighboursBoolean = torch.eq(cIdxs, i)
			featNeighbours = combinedFeatures[detNeighboursBoolean]

			# verfiying that we read all the detections
			detCheck += featNeighbours.shape[0]

			# doing max pooling - across neighbours
			featNeighbours = torch.max(featNeighbours, 0, keepdim=True)[0]

			# concatenating all 
			if (maxPooledFeatures is not None):
				maxPooledFeatures = torch.cat((maxPooledFeatures, featNeighbours))
			else:
				maxPooledFeatures = featNeighbours

		if (detCheck != cIdxs.shape[0]):
			raise Exception("missed detections - detection sizes mismatch")

		# post max-pooling FC layers
		for layer in self.block_pwfeat_postmax_fc_layers:
			maxPooledFeatures = layer(maxPooledFeatures)
	
		# last FC layer
		detFeaturesRefined = self.outputFC(maxPooledFeatures)
		
		outFeatures = detFeatures + detFeaturesRefined
		outFeatures = self.outputReLU(outFeatures)

		return outFeatures


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
		super(GNet, self).__init__()

		self.numClasses = numClasses
		self.neighbourIoU = 0.3	

		# FC layers to generate pairwise features
		self.pwfeatRawInput = 9
		self.pwfeatInnerDim = 256
		self.pwfeatOutDim = 32
		self.num_pwfeat_fc = 3 # keep it >= 3 for now!
		# TODO: weight and bias initialization
		self.pwfeat_gen_layers = []
		self.pwfeat_gen_layers.append(nn.Sequential(
										nn.Linear(self.pwfeatRawInput, self.pwfeatInnerDim, bias=True),
										nn.ReLU(inplace=True)
									))
		for _ in range(self.num_pwfeat_fc-2):
			self.pwfeat_gen_layers.append(nn.Sequential(
											nn.Linear(self.pwfeatInnerDim, self.pwfeatInnerDim, bias=True),
											nn.ReLU(inplace=True)
										))
		self.pwfeat_gen_layers.append(nn.Sequential(
										nn.Linear(self.pwfeatInnerDim, self.pwfeatOutDim, bias=True),
										nn.ReLU(inplace=True)
									))
		self.pwfeat_gen_layers = nn.ModuleList(self.pwfeat_gen_layers)

		# 'block'
		self.shortcutDim = 128
		self.numBlocks = numBlocks
		self.singleBlock = Block()
		self.blockLayers = nn.ModuleList([copy.deepcopy(self.singleBlock) for _ in range(self.numBlocks)])

	def forward(self, data):
		"""
		
		"""
		# since batch size will always remain as 1
		data = data[0]

		# 15000 boxes are too much for now - reducing - change later
		no_detections = 1000

		detScores = data['scores'][:no_detections]
		dtBoxes = data['detections'][:no_detections]
		gtBoxes = data['gt_boxes']		
		
		detScores = torch.from_numpy(detScores)
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

		# finding neighbours of all detections - torch.nonzero() equivalent of tf.where(condition)
		neighbourPairIds = torch.nonzero(torch.ge(dt_dt_iou, self.neighbourIoU))
		pair_c_idxs = neighbourPairIds[:, 0]
		pair_n_idxs = neighbourPairIds[:, 1]

		# generating pairwise features
		pairFeatures = self.generatePairwiseFeatures(pair_c_idxs, pair_n_idxs, neighbourPairIds, detScores, dt_dt_iou, dtBoxesData)
		pairFeatures = pairFeatures.float() # can we fix this?
		pairFeatures = self.pairwiseFeaturesFC(pairFeatures)
		
		numDets = dtBoxes.shape[0]

		# input to the first block must be all zero's
		startingFeatures = torch.zeros([numDets, self.shortcutDim], dtype=torch.float32)
		
		# getting refined features
		detFeatures = startingFeatures
		for layer in self.blockLayers:
			detFeatures = layer(detFeatures, pair_c_idxs, pair_n_idxs, pairFeatures)
		
		# 

	def pairwiseFeaturesFC(self, pairFeatures):
		"""
			Fully connected layers to generate pairwise features
		"""
		for layer in self.pwfeat_gen_layers:
			pairFeatures = layer(pairFeatures)

		return pairFeatures

	@staticmethod
	def generatePairwiseFeatures(c_idxs, n_idxs, neighbourPairIds, detScores, dt_dt_iou, dtBoxesData):
		"""
			Function to compute pairwise features for detections
		"""
		# we don't have multi-class score for detections just the objectness-score

		# getting objectness-score
		cScores = detScores[c_idxs]
		nScores = detScores[n_idxs]

		# gathering ious values between pairs
		ious = dt_dt_iou[neighbourPairIds[:, 0], neighbourPairIds[:, 1]].reshape(-1, 1)

		x1, y1, w, h, _, _, _ = dtBoxesData
		c_w = w[c_idxs]
		c_h = h[c_idxs]
		c_scale = (c_w + c_h) / 2.0
		c_cx = x1[c_idxs] + c_w / 2.0
		c_cy = y1[c_idxs] + c_h / 2.0
		
		n_w = w[n_idxs]
		n_h = h[n_idxs]
		n_cx = x1[n_idxs] + n_w / 2.0
		n_cy = y1[n_idxs] + n_h / 2.0
		
		# normalized x, y distance
		x_dist = torch.sub(n_cx, c_cx)
		y_dist = torch.sub(n_cy, c_cy)
		l2_dist = torch.div(torch.sqrt(x_dist ** 2 + y_dist ** 2), c_scale)
		x_dist /= c_scale
		y_dist /= c_scale

		# scale difference
		log2 = torch.log(torch.Tensor([2.0]))[0]
		w_diff = torch.log(n_w / c_w) / log2
		h_diff = torch.log(n_h / c_h) / log2
		aspect_diff = (torch.log(n_w / n_h) - torch.log(c_w / c_h)) / log2

		# concatenating all properties
		pairFeatures = torch.cat((cScores, nScores, ious, x_dist, y_dist, l2_dist, w_diff, h_diff, aspect_diff), 1)

		return pairFeatures

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