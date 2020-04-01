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
		self.reduced_dim = 32 # move to config file

		# first fully connected layer
		self.fc1 = nn.Sequential(
						nn.Linear((-input_dimension-), self.reduced_dim, bias=True),
						nn.ReLU(inplace=True)
					)

	def forward(self, data):
		"""
			Input
				data: Detections, format?
		"""
		block_fc1_feats = self.fc1(data)

		block_fc1_neighbor_feats = feats

		

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
