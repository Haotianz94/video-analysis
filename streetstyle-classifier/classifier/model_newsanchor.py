import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.inception import model_urls

import numpy as np

class NewsAnchorClassifier(nn.Module):
	'''
	Network to classify style attributes in images. Returns list of size
	num_attributes with classification results for each enabled attribute.
	'''
	def __init__(self):
		super(self.__class__, self).__init__()
		model_urls['inception_v3_google'] = model_urls['inception_v3_google'].replace('https://', 'http://') # hack to download pretrained model w/ python 2.7.6
		self.inception = models.inception_v3(pretrained=True)
		self.inception.aux_logits = False
		feature_size = self.inception.fc.in_features
		# replace fc with dummy layer that just passes through feature vector
		self.inception.fc = PassThrough()
		# whether to evaluate each attribute (all True by default)
		self.attrib_eval_list = [True]*16
		# all attribute layers
		self.clothing_pattern = nn.Linear(feature_size, 6)
		self.major_color = nn.Linear(feature_size, 14)
		self.wearing_necktie = nn.Linear(feature_size, 2)
		self.collar_prescence = nn.Linear(feature_size, 2)
		self.wearing_scarf = nn.Linear(feature_size, 2)
		self.sleeve_length = nn.Linear(feature_size, 3)
		self.neckline_shape = nn.Linear(feature_size, 3)
		self.clothing_category = nn.Linear(feature_size, 7)
		self.wearing_jacket = nn.Linear(feature_size, 2)
		self.wearing_hat = nn.Linear(feature_size, 2)
		self.wearing_glasses = nn.Linear(feature_size, 2)
		self.multiple_layers = nn.Linear(feature_size, 2)
		self.necktie_color = nn.Linear(feature_size, 14)
		self.necktie_pattern = nn.Linear(feature_size, 3)
		self.hair_color = nn.Linear(feature_size, 5)
		self.hair_length = nn.Linear(feature_size, 4)
		# list for forward pass
		self.attrib_layers = [self.clothing_pattern, self.major_color, self.wearing_necktie, self.collar_prescence,
							  self.wearing_scarf, self.sleeve_length, self.neckline_shape, self.clothing_category,
							  self.wearing_jacket, self.wearing_hat, self.wearing_glasses, self.multiple_layers,
							  self.necktie_color, self.necktie_pattern, self.hair_color, self.hair_length]

	def set_eval_attributes(self, attrib_list):
		'''
		Set boolean list to determine which attributes will be classified during evaluation.
		List must be the same size at num_attribs.
		'''
		if len(attrib_list) != len(self.attrib_eval_list):
			raise RuntimeError('Given boolean attribute list size does not match the number of attributes!')
		self.attrib_eval_list = attrib_list

	def forward(self, input):
		feature_vec = self.inception(input)
		out = []
		# TODO is there a way this can be parallelized? very inefficient
		for idx, attrib_layer in enumerate(self.attrib_layers):
			if self.attrib_eval_list[idx]:
				out.append(attrib_layer(feature_vec))
		return out, feature_vec

class PassThrough(nn.Module):
	def __init__(self):
		super(PassThrough, self).__init__()

	def forward(self, input):
		return input