import copy
import time as timer

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import xavierInitialization

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
		self.shortcutDim = 128 # 32
		self.reducedDim = 32 # 16

		# FC layer for input detection - input will be the dimensions for each box
		self.fc1 = nn.Sequential(
						nn.Linear(self.shortcutDim, self.reducedDim, bias=True),
						nn.ReLU(inplace=True)
					)

		# FC layers for pairwise features
		self.num_block_pwfeat_fc = 2 # keep it >=1
		self.blockInnerDim =  2 * 32 # 2 * 16 # why is it defined like this?
		self.block_pwfeat_fc_layers = []
		self.block_pwfeat_fc_layers.append(nn.Sequential(
											nn.Linear(3*32, self.blockInnerDim, bias=True),
											# nn.Linear(3*16, self.blockInnerDim, bias=True),
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

		self.weightInitMethod = 'xavier'
		self.initializeParameters(method=self.weightInitMethod)

	def forward(self, detFeatures, cIdxs, nIdxs, pairFeatures):
		"""
			Input
				detFeatures: RPN detections features, (#detections, 128)
		"""
		block_fc1_feats = self.fc1(detFeatures)
		block_fc1_neighbor_feats = block_fc1_feats
		
		cFeats = block_fc1_feats[cIdxs]
		nFeats = block_fc1_neighbor_feats[nIdxs]

		# zeroing out neighbour features where cIdxs = nIdxs
		isIdRow = torch.eq(cIdxs, nIdxs)
		# zeros = torch.zeros(nFeats.shape, dtype=nFeats.dtype)
		# much faster - https://discuss.pytorch.org/t/creating-tensors-on-gpu-directly/2714
		zeros = torch.cuda.FloatTensor(nFeats.shape).fill_(0)

		nFeats = torch.where(isIdRow.reshape(-1, 1), zeros, nFeats)

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

	def initializeParameters(self, method='xavier'):
		"""
			Initializing weights and bias of all the FC layer
		"""
		if method == 'xavier':
			initializationMethod = xavierInitialization
		else:
			raise Exception("Need to implement other initialization methods")

		# initializing all the layers
		self.fc1.apply(initializationMethod)
		self.block_pwfeat_fc_layers.apply(initializationMethod)
		self.block_pwfeat_postmax_fc_layers.apply(initializationMethod)
		self.outputFC.apply(initializationMethod)



				# # overlaps
				# self.det_anno_iou = self._iou(
				#     self.dets_boxdata, self.gt_boxdata, self.gt_crowd)
				
				# self.det_det_iou = self._iou(self.dets_boxdata, self.dets_boxdata)
				
				# if self.multiclass:
				#     # set overlaps of detection and annotations to 0 if they
				#     # have different classes, so they don't get matched in the
				#     # loss
				#     print('doing multiclass NMS')
				#     same_class = tf.equal(
				#         tf.reshape(self.det_classes, [-1, 1]),
				#         tf.reshape(self.gt_classes, [1, -1]))
				#     zeros = tf.zeros_like(self.det_anno_iou)
					
				#     self.det_anno_iou = tf.select(same_class,
				#                                   self.det_anno_iou, zeros)
					
					
					
	# def _geometry_feats(self, c_idxs, n_idxs):
	#     with tf.variable_scope('pairwise_features'):
	#         if self.multiclass:
	#             mc_score_shape = tf.pack([self.num_dets, self.num_classes])
	#             # classes are one-based (0 is background)
	#             mc_score_idxs = tf.stack(
	#                 [tf.range(self.num_dets), self.det_classes - 1], axis=1)
	#             det_scores = tf.scatter_nd(
	#                 mc_score_idxs, self.det_scores, mc_score_shape)
	#         else:
	#             det_scores = tf.expand_dims(self.det_scores, -1)
				
				
class GNet(nn.Module):
	"""
		'GossipNet' architecture
	"""
	def __init__(self, numBlocks):
		"""
			Initializing the gossipnet architecture
			Input:
				num_classes: Number of classes in the dataset
				num_blocks: Number of blocks to be defined in the network; =4
				class_weights: ?
		"""
		super(GNet, self).__init__()

		self.neighbourIoU = 0.2

		# FC layers to generate pairwise features
		self.pwfeatRawInput = 9
		self.pwfeatInnerDim = 256 # 32
		self.pwfeatOutDim = 32 # 16
		self.num_pwfeat_fc = 3 # keep it >= 3 for now!
		# TODO: weight and bias initialization
		self.pwfeat_gen_layers = []
		self.pwfeat_gen_layers.append(nn.Sequential(
										nn.Linear(self.pwfeatRawInput, self.pwfeatInnerDim, bias=True),
										nn.ReLU(inplace=True)
									))
		for _ in range(self.num_pwfeat_fc-2): # =1
			self.pwfeat_gen_layers.append(nn.Sequential(
											nn.Linear(self.pwfeatInnerDim, self.pwfeatInnerDim, bias=True),
											nn.ReLU(inplace=True)
										))
		self.pwfeat_gen_layers.append(nn.Sequential(
										nn.Linear(self.pwfeatInnerDim, self.pwfeatOutDim, bias=True),
										nn.ReLU(inplace=True)
									))
		self.pwfeat_gen_layers = nn.ModuleList(self.pwfeat_gen_layers)

		# 'block' layers
		self.shortcutDim = 128 # 32
		self.numBlocks = numBlocks
		self.blockLayers = nn.ModuleList([Block() for _ in range(self.numBlocks)])

		# 'fc' layers before generating the updated score
		self.num_score_fc = 3
		self.score_fc_layers = []
		for _ in range(self.num_score_fc):
			self.score_fc_layers.append(nn.Sequential(
											nn.Linear(self.shortcutDim, self.shortcutDim, bias=True),
											nn.ReLU(inplace=True)
										))
		self.score_fc_layers = nn.ModuleList(self.score_fc_layers)

		# new scores - a single (1) score per detection
		self.predictObjectnessScores = nn.Sequential(
									nn.Linear(self.shortcutDim, 1, bias=True),
								)

		# initializing weights and bias of all layers
		self.weightInitMethod = 'xavier'
		self.initializeParameters(method=self.weightInitMethod)

	#def forward(self, batch, no_detections=300, min_score=0.0): #MJ: data is a batch with batch_size =1
	def forward(self, batch):
		"""
			Main computation
		"""
		# # since batch size will always remain as 1
		# data = data[0]
		all_normalized_losses = []
		all_nonnormalized_losses = []
		#all_objectiveness_scores = []

		if len(batch) == 2:
			detection_batch, gt_batch = batch
		else:
			detection_batch = batch
			gt_batch = None

		batch_size = detection_batch.size(dim=0)
		detection_max_length = detection_batch.size(dim=1)

		if not self.training:
			all_objectiveness_scores = torch.ones((batch_size, detection_max_length)) * -1
			if torch.cuda.is_available():
				all_objectiveness_scores = all_objectiveness_scores.cuda()

		for item_id, detections in enumerate(detection_batch):
			detection_boxes = detections[detections[:, 0] != -1, :4]
			detection_classes = detections[detections[:, 0] != -1, 4]
			scores = detections[detections[:, 0] != -1, 5]

			if gt_batch is not None:
				gts = gt_batch[item_id]
				gt_boxes = gts[gts[:, 0] != -1, :4]
				gt_classes = gts[gts[:, 0] != -1, 4]

			#boxes_to_keep = item['scores'] > min_score  #MJ: boxes_to_keep is a boolean array
			# boxes_to_keep = scores > min_score

			# #item['scores'] = item['scores'][boxes_to_keep]
			# #item['detections'] = item['detections'][boxes_to_keep]
			# #item['detection_classes'] = item['detection_classes'][boxes_to_keep]
			# scores = scores[boxes_to_keep]
			# detection_boxes = detection_boxes[boxes_to_keep]
			# detection_classes = detection_classes[boxes_to_keep]
   
			item_dict = {
				'scores': scores,
				'detections': detection_boxes,
				'detection_classes': detection_classes,
			}

			#if 'gt_boxes' not in item:
			if gts is None:
				#objectnessScores = self.compute(item_dict, no_detections)
				objectnessScores = self.compute(item_dict)
			else:
				item_dict['gt_boxes'] = gt_boxes
				item_dict['gt_classes'] = gt_classes

				#losses, objectnessScores = self.compute(item_dict, no_detections)
				losses, objectnessScores = self.compute(item_dict)

				all_normalized_losses.append(losses[0])
				all_nonnormalized_losses.append(losses[1])

			if not self.training:
				#all_objectiveness_scores.append(objectnessScores)
				all_objectiveness_scores[item_id, :objectnessScores.size(dim=0)] = objectnessScores

		#if torch.cuda.is_available():
			#all_objectiveness_scores = torch.stack(all_objectiveness_scores).to(device='cuda')

		if len(all_normalized_losses) == 0 or len(all_nonnormalized_losses) == 0:
			return all_objectiveness_scores

		normalized_loss = sum(all_normalized_losses) / len(all_normalized_losses)
		nonnormalized_loss = sum(all_nonnormalized_losses) / len(all_nonnormalized_losses)

		if self.training:
			return normalized_loss, nonnormalized_loss
		else:
			return normalized_loss, nonnormalized_loss, all_objectiveness_scores

	#def compute(self, data, no_detections):
	def compute(self, data):
		detScores = data['scores']#[:no_detections]  #confidence scores for bbox predictions by beat-fcos. detScores :len=822, say
		dtBoxes = data['detections']#[:no_detections] #bbox predictions by beat-fcos                       dtBoxes : len=822
  
		if 'gt_boxes' in data:
			gtBoxes = data['gt_boxes']		              #annotations for gt boxes                       gtBoxes : len=98
			#gtBoxes = torch.from_numpy(gtBoxes).cuda()
			gtBoxesData = self.getBoxData(gtBoxes)

		multilabel = False
		if 'gt_classes' in data and 'detection_classes' in data:
			multilabel = True

			gt_classes = data['gt_classes']
			detection_classes = data['detection_classes']#[:no_detections]

			if isinstance(detScores, np.ndarray):
				gt_classes = torch.from_numpy(gt_classes).type(torch.cuda.FloatTensor)
	
			if isinstance(dtBoxes, np.ndarray):
				detection_classes = torch.from_numpy(detection_classes).type(torch.cuda.FloatTensor)

		if isinstance(detScores, np.ndarray):
			detScores = torch.from_numpy(detScores).type(torch.cuda.FloatTensor)
   
		if isinstance(dtBoxes, np.ndarray):
			dtBoxes = torch.from_numpy(dtBoxes).cuda()

		# getting box information (x1, y1, w, h, x2, y2, area) from (x1,y1,x2,y2)
		dtBoxesData = self.getBoxData(dtBoxes)   #MJ: return (x1, y1, width, height, x2, y2, area)
		# gtBoxesData = self.getBoxData(gtBoxes) #MJ: dtBoxesData[0].shape = (822,1)

		# computing IoU matrix between detections and detections: 
		# MJ: It is an association matrix between detections for deciding which detections are neighbors, i.e,
		# the detections that may point to the same object.
		#
		dt_dt_iou = self.iou(dtBoxesData, dtBoxesData)  #MJ: dt_dt_iou matrix: shape = (822,822), say;from PIL import Imageim = Image.fromarray(np.uint8(mat))

		# we don't have classes for detections - just the gt_classes
		# so doing single class nms - discuss on this! - okay!

		##############################################################################
		#MJ: Make the dt_dt_iou between beat and downbeat predictions zero.
		
		if multilabel:
			dt_dt_same_class = torch.eq(detection_classes.reshape(-1, 1), detection_classes.reshape(1, -1))
			dt_dt_iou = torch.where(
				dt_dt_same_class,
				dt_dt_iou,
				torch.zeros(dt_dt_same_class.shape).to(dt_dt_same_class.device)
			)

		# config 0.
		# finding neighbours of all detections - torch.nonzero() equivalent of tf.where(condition)
		neighbourPairIds = torch.nonzero(torch.ge(dt_dt_iou, self.neighbourIoU))  #MJ: self.neighbourIoU = 0.2; neighbourPairIds: shape =(23082,2),2:(i,j), from 822x822=675,684
        #MJ: returns a 2-D tensor where each row is the index (i,j) for a nonzero value. (N,1) or (N,2)
        
		# code to get number of neighbour pairs
		# self.no_neighbour = len(neighbourPairIds)

		# print ("No detections: {}, no pairs: {}".format(dtBoxes.shape[0], neighbourPairIds.shape[0]))

		# config 1. limiting to top 50 IoU matches - nieghbourIoU not used
		# args = torch.argsort(dt_dt_iou, descending=True)
		# neighbourPairIds = torch.nonzero(torch.le(args, 49))

		# config 2. using clustering centers
		# same as the original
		# neighbourPairIds = torch.nonzero(torch.ge(dt_dt_iou, self.neighbourIoU))

		# config 3. using neighbours from the cluster centers
		# masking non-cluster detections, IoU = 10
		# cluster_labels = torch.from_numpy(data['cluster_labels']).cuda()
		# mask = torch.eq(cluster_labels, cluster_labels.view(-1, 1))
		# dt_dt_iou[mask] = 10
		# # sorting so that non-cluster detections are at the end 
		# # masking again so that torch
		# args = torch.argsort(dt_dt_iou, descending=False)
		# args[mask] = 100000 
		# neighbourPairIds = torch.nonzero(torch.le(args, 49))

		pair_c_idxs = neighbourPairIds[:, 0]  # pair_c_idxs may contain repeated indices
		pair_n_idxs = neighbourPairIds[:, 1]  # pair_n_idxs may contain repeated indices

		# model-check code - used in train.py
		self.neighbourPairIds = neighbourPairIds
		# print ("Number of neighbours being processed: {}".format(len(neighbourPairIds)))

		# 
		# generate handcrafted pairwise features, which are used to compute objectnessScores = self.predictObjectnessScores(detFeatures)
		pairFeaturesDescriptors =  self.generatePairwiseFeatures(pair_c_idxs, pair_n_idxs, neighbourPairIds, detScores, dt_dt_iou, dtBoxesData) 
		pairFeatures = self.pairwiseFeaturesFC(pairFeaturesDescriptors)  #MJ: pairFeaturesDescriptors:  shape = (23082,9) 
  
		# MJ: self.pairwiseFeaturesFC extract abstract feature maps from the feature descriptors of the detection pairs
		# Fully connected layers to generate pairwise features:
		# 
		# for layer in self.pwfeat_gen_layers:
		# 	pairFeatures = layer(pairFeatures)

		# return pairFeatures

		numDets = dtBoxes.shape[0]  #MJ= 822

		# input to the first block must be all zero's
		# much faster - https://discuss.pytorch.org/t/creating-tensors-on-gpu-directly/2714
		# startingFeatures = torch.zeros([numDets, self.shortcutDim], dtype=torch.float32).cuda()
		startingFeatures = torch.cuda.FloatTensor(numDets, self.shortcutDim).fill_(0)

		# getting refined features:
  
	#MJ: 
	# Detection features. The blocks of our network take the detection feature vector of each detection as input and outputs
	# an updated vector (see high-level illustration in figure 2). Outputs from one block are input to the next one. The
	# values inside this c = 128 dimensional feature vector are learned implicitly during the training. 
	# The output of the last block is used to generate the new detection score for each detection
		detFeatures = startingFeatures
		for layer in self.blockLayers:  # MJ: self.blockLayers = nn.ModuleList([Block() for _ in range(self.numBlocks)])
			detFeatures = layer(detFeatures, pair_c_idxs, pair_n_idxs, pairFeatures) #MJ: pairFeatures is fed into each block repeatedly

		# passing through scoring FC layers
		for layer in self.score_fc_layers:
			detFeatures = layer(detFeatures)

		# predicting new scores: objectnessScores: shape = (822,)
		objectnessScores = self.predictObjectnessScores(detFeatures)  #MJ: detFeatures: shape =(822, 128), 128-dim vector for 822 detections
		 # objectnessScores = f(x_p) in Eq (1) in https://arxiv.org/pdf/1511.06437.pdf
		 # and = s_i
  #MJ: # new scores - a single (1) score per detection
		# self.predictObjectnessScores = nn.Sequential(
		# 							nn.Linear(self.shortcutDim, 1, bias=True),
		# 						)
  
		objectnessScores = objectnessScores.reshape(-1)  #MJ: objectnessScore: shape = (822,1)

		# # test mode should return from here
		if not self.training and 'gt_boxes' not in data:
			return objectnessScores

		# computing IoU between detections and ground truth
		dt_gt_iou = self.iou(dtBoxesData, gtBoxesData)  #MJ: dt_gt_iou: shape = (822,98)
  
		#################################################################
		#MJ: make dt_gt_iou between beat and downbeat zero.

		if multilabel:
			dt_gt_same_class = torch.eq(detection_classes.reshape(-1, 1), gt_classes.reshape(1, -1))
			dt_gt_iou = torch.where(
				dt_gt_same_class,
				dt_gt_iou,
				torch.zeros(dt_gt_same_class.shape).to(dt_gt_same_class.device)
			)

		### label matching for training
		# original implementation works on COCO dataset and has 'gt_crowd' detections
		# 'gt_crowd' detections are handled in a different way - making the logic complicated
		# since we are using VRD (and VG later) datasets, our logic doesn't have to be complicated
		# labels, dt_gt_matching = self.dtGtMatching(dt_gt_iou, objectnessScores)
		# start = timer.time()
		labels, _ = self.dtGtMatching(dt_gt_iou, objectnessScores) # #objectnessScores: objectnessScores of all detections thru gnet. 
		#MJ: objectnessScores: Recomputed scores for the detections
		# The output of self.dtGtMatching(dt_gt_iou, objectnessScores):
		#   labels: Boolean tensor representing which detections/samples are to be treated as true positives
		#   dt_gt_matching: which detection gets matched to which gt
		# print (timer.time() - start)

		### computing sample losses
		# equivalent 'tf.nn.sigmoid_cross_entropy_with_logits' -> 'torch.nn.BCEWithLogitsLoss'
  
		sampleLossFunction = nn.BCEWithLogitsLoss(weight=None, reduction='none')
  #MJ: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
  
		sampleLosses = sampleLossFunction(objectnessScores, labels)  #MJ: labels = targets: their values should be 0 or 1
				   #MJ: labels: Boolean tensor representing which detections/samples are to be treated as true positives
				   # loss(objectnessScores, labels) = list( labels[i]*log (sigmoid(objectnessScores[i])) + (1-labels[i] * log( 1- sigmoid(objectnessScores[i])))

		lossNormalized = torch.mean(sampleLosses)
		lossUnnormalized = torch.sum(sampleLosses)

		return (lossNormalized, lossUnnormalized), objectnessScores

	def initializeParameters(self, method='xavier'):
		"""
			Initializing weights and bias of all the FC layer
		"""
		if method == 'xavier':
			initializationMethod = xavierInitialization
		else:
			raise Exception("Need to implement other initialization methods")

		# initializing all the layers
		self.pwfeat_gen_layers.apply(initializationMethod)
		self.score_fc_layers.apply(initializationMethod)
		self.predictObjectnessScores.apply(initializationMethod)

	@staticmethod
	def dtGtMatching(dt_gt_iou, objectnessScores, iouThresh=0.5):
		"""
			Matching detections with ground truth labels using the recomputed objectness score, each gt is matched with 
			exactly one detection
			Input:
				dt_gt_iou: IoU between detections and ground truth bbox's
				objectnessScores: Recomputed scores for the detections
				iouThresh: iou-threshold for the detections to be considered as positives
			Return:
				labels: Boolean tensor representing which detections are to be treated as true positives
				dt_gt_matching: which detection gets matched to which gt: The gt box matched to each detection (whose num is 822,say)
		"""
		# sorting objectness score - getting their index's 
		objectnessScores = objectnessScores.reshape(-1)   #
		sortedIndexs = torch.argsort(objectnessScores, descending=True)

		numDts = dt_gt_iou.shape[0]
		numGts = dt_gt_iou.shape[1]

		# each gt need to be matched exactly once
		# isGtMatched = torch.zeros(numGts, dtype=torch.int32)
		isGtMatched = torch.cuda.IntTensor(numGts).fill_(0)     #MJ: isGtMatched: shape = (98,)

		# labels = torch.zeros(numDts, dtype=torch.float32)
		labels = torch.cuda.FloatTensor(numDts).fill_(0)
		# dt_gt_matching = torch.zeros(numDts, dtype=torch.int32)
		# dt_gt_matching.fill_(-1)
		dt_gt_matching = torch.cuda.IntTensor(numDts).fill_(-1)  #MJ: dt_gt_matching: shape =(822,)

		for i in range(numDts): #MJ: 822
			dtIndex = sortedIndexs[i]  #MJ: sortedIndexs from objectnessScores
			iou = iouThresh
			match = -1
			# match2 = -1
   
			dt_gt_iou_unmatched = torch.where(
				isGtMatched != 1,
				dt_gt_iou[dtIndex],
				torch.zeros(isGtMatched.shape).to(isGtMatched.device) * -1
			)

			iou = torch.max(dt_gt_iou_unmatched)

			if iou < iouThresh:
				iou = iouThresh
			else:
				match = torch.argmax(dt_gt_iou_unmatched)
			# print(1, iou2, match2)

			# for gtIndex in range(numGts):  #MJ: 98
			# 	# is gt already matched
			# 	if isGtMatched[gtIndex] == 1:
			# 		continue

			# 	# continue until we get a better detection
			# 	if dt_gt_iou[dtIndex, gtIndex] < iou:
			# 		continue
				
			# 	# store the best detection
			# 	iou = dt_gt_iou[dtIndex, gtIndex]  #MJ: dtIndex=460; gtIndex=85, say; iou = 0.7792
			# 	match = gtIndex
			# print(2, iou, match)
			if (match > -1):
				isGtMatched[match] = 1  #MJ: gt box at index match is matched to detection at index dtIndex
				labels[dtIndex] = 1.
				dt_gt_matching[dtIndex] = match

		# print (isGtMatched)

		return (labels, dt_gt_matching) #MJ: if labels[i]=1, it means that detection i is the only positive anchor which predicts the target object at index match

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

		# getting objectness-score: detScores: shape = torch.Size([822]); c_idxs, n_idxs: torch.Size([23082])
		cScores = detScores[c_idxs].unsqueeze(dim=1) #MJ: cScores: shape = (23082,1)
		nScores = detScores[n_idxs].unsqueeze(dim=1)  #MJ: nScores: shape = (23082,1)

		# gathering/selecting ious values between pairs: dt_dt_iou: shape=(822,822)
		ious = dt_dt_iou[neighbourPairIds[:, 0], neighbourPairIds[:, 1]].reshape(-1, 1) #shape=torch.Size([23082, 1])

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

		# concatenating all properties of each neighboring detection
		pairFeatures = torch.cat((cScores, nScores, ious, x_dist, y_dist, l2_dist, w_diff, h_diff, aspect_diff), 1)

		return pairFeatures #MJ: shape = (23082,9)

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

		width = torch.max(torch.cuda.FloatTensor([0.0]), torch.sub(x2, x1))
		height = torch.max(torch.cuda.FloatTensor([0.0]), torch.sub(y2, y1))

		intersection = torch.mul(width, height)

		return intersection

	@staticmethod
	def iou(boxes1, boxes2):
		"""
			Compute IoU values between boxes1 and boxes2;
		   boxes1, boxes2: (x1, y1, width, height, x2, y2, area)
		"""
		area1 = boxes1[6].reshape(-1, 1) # area1 set in rows: 1xN
		area2 = boxes2[6].reshape(1, -1) # area2 set in columns: Mx1

		intersection = 	GNet.intersection(boxes1, boxes2)
		union = torch.sub(torch.add(area1, area2), intersection)
		iou = torch.div(intersection, union) #MJ: iou = NxM overlap matrix between detectors and gt boxes

		return iou

	@staticmethod
	def getBoxData(boxes):
		"""
			Getting box information (x1, y1, w, h, x2, y2, area) from the original format (x1, y1, x2, y2)
		"""
		x1 = boxes[:, 0].reshape(-1, 1).type(torch.cuda.FloatTensor)
		y1 = boxes[:, 1].reshape(-1, 1).type(torch.cuda.FloatTensor)
		x2 = boxes[:, 2].reshape(-1, 1).type(torch.cuda.FloatTensor)
		y2 = boxes[:, 3].reshape(-1, 1).type(torch.cuda.FloatTensor)

		width = x2 - x1
		height = y2 - y1

		# find a torch equivalent or convert to numpy and then check
		# assert np.all(width > 0) and np.all(height > 0), "rpn boxes are incorrect - height or width is negative"

		area = torch.mul(width, height)

		return (x1, y1, width, height, x2, y2, area)