from base_test import BaseTest
from dataset_newsanchor_infer import NewsAnchorDataset
from model_newsanchor import NewsAnchorClassifier

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy
import pickle

import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms
from data_utils import ResizeTransform

class NewsAnchorClassifierInfer(BaseTest):

    def __init__(self, use_gpu=True):
        super(self.__class__, self).__init__(use_gpu)

    def create_data_loaders(self, img_data_dir, img_manifest_path, batch_size):
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = NewsAnchorDataset(img_data_dir, img_manifest_path, batch_size=batch_size, 
                                        transform=transform)
        self.infer_loader = dataset

    def imshow(self, inp):
        """Imshow for Tensor."""
        plt.figure()
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)

    def visualize_single_batch(self):
        # get some random training images
        images, manifest = self.infer_loader.next_infer()
        self.infer_loader.init()
        img = torchvision.utils.make_grid(images[:16], nrow=4)
        self.imshow(img)

    def load_model(self, checkpoint_name):
        self.model = NewsAnchorClassifier()
        self.model.load_state_dict(torch.load(checkpoint_name, map_location=lambda storage, loc: storage))
        if self.use_gpu:
            self.model = self.model.cuda()
            
    def infer(self):
        '''
        Use the best model to infer entire dataset.
        Returns infer result for each attribute.
        '''
        self.model.eval()
        infer_result = []
        feature_result = []
        print("Inference starting...")
        iter_count = 0
        # get first batch
        images, manifest = self.infer_loader.next_infer()
        start_time = time.time()
        while images is not None:
            print("Infer iteration %d, time %f" % (iter_count, time.time()-start_time))
            
            if self.use_gpu:
                images = Variable(images.float().cuda(), requires_grad=False)
            else:
                images = Variable(images.float(), requires_grad=False)

            # classify mini-batch
            output, feature = self.model(images)
            if self.use_gpu:
                feature = feature.cpu().data.numpy()
            else:
                feature = feature.data.numpy()
            attribute_pre = np.zeros((len(images), len(output)))
            # store results
            for j, attrib_output in enumerate(output):
                _, predicted = torch.max(attrib_output, 1)
#                 video_name = manifest[j][0]
#                 pid = manifest[j][1]
#                 if not video_name in infer_result:
#                     infer_result[video_name] = []
#                 if len(infer_result[video_name]) < pid:
#                     infer_result[video_name].append([])
#                 infer_result[video_name][pid].append(predicted)
                if self.use_gpu:
                    attribute_pre[:, j] = predicted.cpu().data.numpy()
                else:
                    attribute_pre[:, j] = predicted.data.numpy()
            for i, meta in enumerate(manifest):
                res = copy.deepcopy(meta)
                res.append(attribute_pre[i].tolist())
                infer_result.append(res)
                feature_result.append((res[2], feature[i]))
            
            iter_count += 1
            # next batch
            del images
            del manifest
            del predicted
            del attribute_pre
            images, manifest = self.infer_loader.next_infer()
            
#             if iter_count % 1000 == 0:
#                 pickle.dump(infer_result, open('../../data/cloth/cloth_all_infer.pkl', 'wb'), protocol=2)
#                 pickle.dump(feature_result, open('../../data/cloth/cloth_all_feature.pkl', 'wb'), protocol=2)

        return infer_result, feature_result
