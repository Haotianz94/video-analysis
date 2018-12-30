import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np

class HairStyleClassifier(nn.Module):
    '''
    Network to classify style attributes in images. Returns list of size
    num_attributes with classification results for each enabled attribute.
    '''
    def __init__(self, model_type="inception_v3"):
        super(self.__class__, self).__init__()
        if model_type == "inception_v3":
            self.model = models.inception_v3(pretrained=True)
        elif model_type == "resnet50":
            self.model = models.resnet50(pretrained=True)
        elif model_type == "resnet18":
            self.model = models.resnet18(pretrained=True)
        elif model_type == "vgg16_bn":
            self.model = models.vgg16_bn(pretrained=True)
            
        self.model.aux_logits = False
        feature_size = self.model.fc.in_features
        # replace fc with dummy layer that just passes through feature vector
        self.model.fc = PassThrough()
        
        # whether to evaluate each attribute (all True by default)
        self.attrib_eval_list = [True] * 3
        # all attribute layers
        self.hair_color3 = nn.Linear(feature_size, 3)
        self.hair_color5 = nn.Linear(feature_size, 5)
        self.hair_length = nn.Linear(feature_size, 4)
        # list for forward pass
        self.attrib_layers = [self.hair_color3, self.hair_color5, self.hair_length]

    def set_eval_attributes(self, attrib_list):
        '''
        Set boolean list to determine which attributes will be classified during evaluation.
        List must be the same size at num_attribs.
        '''
        if len(attrib_list) != len(self.attrib_eval_list):
            raise RuntimeError('Given boolean attribute list size does not match the number of attributes!')
        self.attrib_eval_list = attrib_list

    def forward(self, input):
        feature_vec = self.model(input)
        out = []
        # TODO is there a way this can be parallelized? very inefficient
        for idx, attrib_layer in enumerate(self.attrib_layers):
            if self.attrib_eval_list[idx]:
                out.append(attrib_layer(feature_vec))
        return out

class PassThrough(nn.Module):
    def __init__(self):
        super(PassThrough, self).__init__()

    def forward(self, input):
        return input