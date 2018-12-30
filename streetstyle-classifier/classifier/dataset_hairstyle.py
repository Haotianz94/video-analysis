import os
from torch.utils.data import Dataset, DataLoader
import csv
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
import random
import pickle


class HairStyleDataset(object):
    '''
    Handles loading and stratified sampling from the NewsAnchor dataset. Only has training and test splits right now.
    '''
    attributes = [
        {'black' : 0, 'white': 1, 'blond' : 2}, # hair_color3
        {'black' : 0, 'white': 1, 'blond' : 2, 'brown' : 3, 'gray' : 4}, # hair_color5
        {'long' : 0, 'medium' : 1, 'short' : 2, 'bald' : 3} # hair_length
    ]
    def __init__(self, image_data_dir, manifest_data_path, batch_size=32, transform=None,
                test_split_size=15):
        '''
        - image_data_dir: directory holding the image data structure
        - manifest_data_path: file path holding the manifest file
        - batch_size: mini-batch size for sampling
        - transform: torchvision transforms to apply to each image
        - test_split_size: the percentage of people in dataset to use for test split
        '''
        self.image_data_dir = image_data_dir
        self.manifest_data_path = manifest_data_path
        self.batch_size = batch_size
        self.transform = transform
        self.test_split_frac = test_split_size / 100.0
        self.train_split_frac = 1 - self.test_split_frac
        # state variables for mini-batch sampling
        self.cur_attrib = 0
        self.cur_test_start_idx = 0
        # loaded necessary data
        self.preload()

    '''
    Performs necessary precomputation like loading manifest file information,
    and building structure for minibatch sampling.
    '''
    def preload(self):
        # load in file name for images of each anchor
        manifest_data = pickle.load(open(self.manifest_data_path, 'rb'))
        self.img_name_data = []
        self.attrib_data = []
        train_max = 0
        
        # manifest: [(name, attrib)]
        for idx, data in enumerate(manifest_data):
            # for training and evaluation
            if 1. * idx / len(manifest_data) < 1 - self.test_split_frac:
                self.img_name_data.append(data[0])
                self.attrib_data.append(data[1])
            # for test
            else:
                if train_max == 0:
                    train_max = len(self.img_name_data) + 1
                self.img_name_data.append(data[0])
                self.attrib_data.append(data[1])

        print("Total images: ", len(self.img_name_data))
        self.num_images = len(self.img_name_data)
   
        # create splits
        self.train_inds = range(0, train_max)
        self.test_inds = range(train_max, self.num_images)

        # print(self.attribute_data)
        # print(self.img_name_data)

        # build sampling structure:
        # each attribute has a dictionary pointing to all image indices which contain that attribute value
        self.attrib_inds = []
        for attrib in HairStyleDataset.attributes:
            self.attrib_inds.append({x : [] for x in range(0, len(attrib))})
        # fill with training image attributes
        for train_idx in self.train_inds:
            attribs = self.attrib_data[train_idx]
            for attrib_idx, attrib_value in enumerate(attribs):
                if attrib_value != -1:
                    self.attrib_inds[attrib_idx][attrib_value].append(train_idx)

    def next_train(self):
        '''
        Returns the next training mini-batch using stratified sampling.
        '''
        labels = np.zeros((self.batch_size, len(HairStyleDataset.attributes)))
        images = None
        for i in range(0, self.batch_size):
            # sample a random value to use for current attribute
            val = random.randint(0, len(HairStyleDataset.attributes[self.cur_attrib])-1)
            while (len(self.attrib_inds[self.cur_attrib][val]) == 0):
                # want to make sure we actually have data for this value
                val = random.randint(0, len(HairStyleDataset.attributes[self.cur_attrib])-1)
            # get a random image index with that value
            img_idx = self.attrib_inds[self.cur_attrib][val][random.randint(0, len(self.attrib_inds[self.cur_attrib][val])-1)]
            # update
            self.cur_attrib = (self.cur_attrib + 1) % len(HairStyleDataset.attributes)
            # load the image
            img = self.load_img(img_idx)
            label = np.array(self.attrib_data[img_idx])
            # add image and label to batch
            if images is None:
                images = torch.Tensor(self.batch_size, img.shape[0], img.shape[1], img.shape[2])
            images[i] = img
            labels[i] = label
        return images, torch.from_numpy(labels)

    def next_test(self):
        '''
        Returns the next mini-batch from the test split or None if no
        more test data is available. If None, resets so next time called will get the
        first test batch and so on.
        '''
        if self.cur_test_start_idx == len(self.test_inds):
            self.cur_test_start_idx = 0
            return None, None

        images = None
        # start at cur eval start idx
        start_idx = self.cur_test_start_idx
        end_idx = self.cur_test_start_idx + self.batch_size
        # make sure doesn't go past end
        end_idx = end_idx if end_idx <= len(self.test_inds) else len(self.test_inds)
        self.cur_test_start_idx = end_idx
        img_indices = self.test_inds[start_idx:end_idx]
        labels = np.zeros((len(img_indices), len(HairStyleDataset.attributes)))
        for i, img_idx in enumerate(img_indices):
            # load the image
            img = self.load_img(img_idx)
            label = np.array(self.attrib_data[img_idx])
            if images is None:
                images = torch.Tensor(len(img_indices), img.shape[0], img.shape[1], img.shape[2])
            images[i] = img
            labels[i] = label
        return images, torch.from_numpy(labels)    
   

    def load_img(self, img_idx):
        '''
        Loads the given image with corresponding crop and transform applied.
        '''
        img_name = self.img_name_data[img_idx]
        # load the image
        img_path = os.path.join(self.image_data_dir, img_name)
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img