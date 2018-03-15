import os
from torch.utils.data import Dataset, DataLoader
import csv
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
import random


class StreetStyleDataset(object):
    '''
    Handles loading and stratified sampling from the StreetStyle-27k dataset.
    '''
    attributes = [
        {'Solid': 0, 'Graphics' : 1, 'Striped' : 2, 'Floral' : 3, 'Plaid' : 4, 'Spotted' : 5}, # clothing_pattern
        {'Black' : 0, 'White' : 1, 'More than 1 color' : 2, 'Blue' : 3, 'Gray' : 4, 'Red' : 5,
                'Pink' : 6, 'Green' : 7, 'Yellow' : 8, 'Brown' : 9, 'Purple' : 10, 'Orange' : 11,
                'Cyan' : 12}, # major_color
        {'No': 0, 'Yes' : 1}, # wearing_necktie
        {'No': 0, 'Yes' : 1}, # collar_presence
        {'No': 0, 'Yes' : 1}, # wearing_scarf
        {'Long sleeve' : 0, 'Short sleeve' : 1, 'No sleeve' : 2}, # sleeve_length
        {'Round' : 0, 'Folded' : 1, 'V-shape' : 2}, # neckline_shape
        {'Shirt' : 0, 'Outerwear' : 1, 'T-shirt' : 2, 'Dress' : 3,
            'Tank top' : 4, 'Suit' : 5, 'Sweater' : 6}, # clothing_category
        {'No': 0, 'Yes' : 1}, # wearing_jacket
        {'No': 0, 'Yes' : 1}, # wearing_hat
        {'No': 0, 'Yes' : 1}, # wearing_glasses
        {'One layer': 0, 'Multiple layers' : 1} # multiple_layers
    ]

    def __init__(self, image_data_dir, manifest_data_dir, batch_size=32, transform=None,
                    train_split_size=80, eval_split_size=10):
        '''
        - image_data_dir: directory holding the image data structure
        - manifest_data_dir: directory holding the streetstyle27k.manifest file
        - batch_size: mini-batch size for sampling
        - transform: torchvision transforms to apply to each image
        - train_split_size: percentage of dataset to use for training
        - eval_split_size: percentage of dataset to use for eval (remaining after train and eval is test)
        '''
        self.image_data_dir = image_data_dir
        self.manifest_data_dir = manifest_data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.train_split_frac = train_split_size / 100.0
        self.eval_split_frac = eval_split_size / 100.0
        # state variables for mini-batch sampling
        self.cur_attrib = 0
        self.cur_eval_start_idx = 0
        self.cur_test_start_idx = 0
        # loaded necessary data
        self.preload()

    '''
    Performs necessary precomputation like loading manifest file information,
    creating split indices, and building structure for minibatch sampling.
    '''
    def preload(self):
        # load in file name, bounding box, and attribute values for each image
        self.bbox_data = []
        self.attrib_data = []
        self.img_name_data = []
        with open(os.path.join(self.manifest_data_dir, 'streetstyle27k.manifest')) as manifest_file:
            reader = csv.reader(manifest_file, delimiter=',')
            next(reader) # header row
            for row in reader:
                # image name
                name = row[1].split('/')[-1]
                self.img_name_data.append(name)
                # bounding box
                bbox_str = row[7:11]
                bbox = [int(x) for x in bbox_str]
                self.bbox_data.append(bbox)
                # attribute values
                attribs_str = row[13:]
                attribs = [-1]*len(StreetStyleDataset.attributes)
                for i, attrib in enumerate(attribs_str):
                    if attrib != '':
                        attribs[i] = StreetStyleDataset.attributes[i][attrib]
                self.attrib_data.append(attribs)

        self.num_images = len(self.bbox_data)
        # create splits
        train_max = int(self.num_images*self.train_split_frac) + 1
        self.train_inds = range(0, train_max)
        eval_max = train_max + int(self.num_images*self.eval_split_frac) + 1
        self.eval_inds = range(train_max, eval_max)
        self.test_inds = range(eval_max, self.num_images)

        # build sampling structure:
        # each attribute has a dictionary pointing to all image indices which contain that attribute value
        self.attrib_inds = []
        for attrib in StreetStyleDataset.attributes:
            self.attrib_inds.append({x : [] for x in range(0, len(attrib))})
        # fill with training image attributes
        for train_idx in self.train_inds:
            attribs = self.attrib_data[train_idx]
            for attrib_idx, attrib_value in enumerate(attribs):
                if attrib_value != -1:
                    self.attrib_inds[attrib_idx][attrib_value].append(train_idx)

        # print(len(self.attrib_inds[9][1])) # num wearing hat
        # print(len(self.attrib_inds[6][2])) # num wearing vneck

    def next_train(self):
        '''
        Returns the next training mini-batch using stratified sampling.
        '''
        labels = np.zeros((self.batch_size, len(StreetStyleDataset.attributes)))
        images = None
        for i in range(0, self.batch_size):
            # sample a random value to use for current attribute
            val = random.randint(0, len(StreetStyleDataset.attributes[self.cur_attrib])-1)
            # get a random image index with that value
            img_idx = self.attrib_inds[self.cur_attrib][val][random.randint(0, len(self.attrib_inds[self.cur_attrib][val])-1)]
            # update
            self.cur_attrib = (self.cur_attrib + 1) % len(StreetStyleDataset.attributes)
            # load the image
            img = self.load_img(img_idx)
            label = np.array(self.attrib_data[img_idx])
            # add image and label to batch
            if images is None:
                images = torch.Tensor(self.batch_size, img.shape[0], img.shape[1], img.shape[2])
            images[i] = img
            labels[i] = label
        return images, torch.from_numpy(labels)

    def next_eval(self):
        '''
        Returns the next mini-batch from the evaluation split or None if no
        more eval data is available. If None, resets so next time called will get the
        first eval batch and so on.
        '''
        if self.cur_eval_start_idx == len(self.eval_inds):
            self.cur_eval_start_idx = 0
            return None, None

        images = None
        # start at cur eval start idx
        start_idx = self.cur_eval_start_idx
        end_idx = self.cur_eval_start_idx + self.batch_size
        # make sure doesn't go past end
        end_idx = end_idx if end_idx <= len(self.eval_inds) else len(self.eval_inds)
        self.cur_eval_start_idx = end_idx
        img_indices = self.eval_inds[start_idx:end_idx]
        labels = np.zeros((len(img_indices), len(StreetStyleDataset.attributes)))
        for i, img_idx in enumerate(img_indices):
            # load the image
            img = self.load_img(img_idx)
            label = np.array(self.attrib_data[img_idx])
            if images is None:
                images = torch.Tensor(len(img_indices), img.shape[0], img.shape[1], img.shape[2])
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
        labels = np.zeros((len(img_indices), len(StreetStyleDataset.attributes)))
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
        # load the image (located in path a/b/c/abcimagename.jpg)
        img_path = os.path.join(self.image_data_dir, img_name[0], img_name[1], img_name[2], img_name)
        img = Image.open(img_path)
        # crop w/ bounding box
        x1, y1, x2, y2 = self.bbox_data[img_idx]
        img = img.crop((x1, y1, x2, y2))
        if self.transform is not None:
            img = self.transform(img)
        return img

### EQUAL SAMPLING VERSION
# import os
# from torch.utils.data import Dataset, DataLoader
# import csv
# from PIL import Image
# import numpy as np
# import torch
# from torch.autograd import Variable
# import random


# class StreetStyleDataset(object):
#     '''
#     Handles loading and stratified sampling from the StreetStyle-27k dataset.
#     '''
#     attributes = [
#         {'Solid': 0, 'Graphics' : 1, 'Striped' : 2, 'Floral' : 3, 'Plaid' : 4, 'Spotted' : 5}, # clothing_pattern
#         {'Black' : 0, 'White' : 1, 'More than 1 color' : 2, 'Blue' : 3, 'Gray' : 4, 'Red' : 5,
#                 'Pink' : 6, 'Green' : 7, 'Yellow' : 8, 'Brown' : 9, 'Purple' : 10, 'Orange' : 11,
#                 'Cyan' : 12}, # major_color
#         {'No': 0, 'Yes' : 1}, # wearing_necktie
#         {'No': 0, 'Yes' : 1}, # collar_presence
#         {'No': 0, 'Yes' : 1}, # wearing_scarf
#         {'Long sleeve' : 0, 'Short sleeve' : 1, 'No sleeve' : 2}, # sleeve_length
#         {'Round' : 0, 'Folded' : 1, 'V-shape' : 2}, # neckline_shape
#         {'Shirt' : 0, 'Outerwear' : 1, 'T-shirt' : 2, 'Dress' : 3,
#             'Tank top' : 4, 'Suit' : 5, 'Sweater' : 6}, # clothing_category
#         {'No': 0, 'Yes' : 1}, # wearing_jacket
#         {'No': 0, 'Yes' : 1}, # wearing_hat
#         {'No': 0, 'Yes' : 1}, # wearing_glasses
#         {'One layer': 0, 'Multiple layers' : 1} # multiple_layers
#     ]

#     def __init__(self, image_data_dir, manifest_data_dir, batch_size=32, transform=None,
#                     train_split_size=80, eval_split_size=10):
#         '''
#         - image_data_dir: directory holding the image data structure
#         - manifest_data_dir: directory holding the streetstyle27k.manifest file
#         - batch_size: mini-batch size for sampling
#         - transform: torchvision transforms to apply to each image
#         - train_split_size: percentage of dataset to use for training
#         - eval_split_size: percentage of dataset to use for eval (remaining after train and eval is test)
#         '''
#         self.image_data_dir = image_data_dir
#         self.manifest_data_dir = manifest_data_dir
#         self.batch_size = batch_size
#         self.transform = transform
#         self.train_split_frac = train_split_size / 100.0
#         self.eval_split_frac = eval_split_size / 100.0
#         self.test_split_frac = 1.0 - self.train_split_frac - self.eval_split_frac
#         # state variables for mini-batch sampling
#         self.cur_attrib = 0
#         self.cur_eval_start_idx = 0
#         self.cur_test_start_idx = 0
#         # loaded necessary data
#         self.preload()

#     '''
#     Performs necessary precomputation like loading manifest file information,
#     creating split indices, and building structure for minibatch sampling.
#     '''
#     def preload(self):
#         # load in file name, bounding box, and attribute values for each image
#         self.bbox_data = []
#         self.attrib_data = []
#         self.img_name_data = []
#         with open(os.path.join(self.manifest_data_dir, 'streetstyle27k.manifest')) as manifest_file:
#             reader = csv.reader(manifest_file, delimiter=',')
#             next(reader) # header row
#             for row in reader:
#                 # image name
#                 name = row[1].split('/')[-1]
#                 self.img_name_data.append(name)
#                 # bounding box
#                 bbox_str = row[7:11]
#                 bbox = [int(x) for x in bbox_str]
#                 self.bbox_data.append(bbox)
#                 # attribute values
#                 attribs_str = row[13:]
#                 attribs = [-1]*len(StreetStyleDataset.attributes)
#                 for i, attrib in enumerate(attribs_str):
#                     if attrib != '':
#                         attribs[i] = StreetStyleDataset.attributes[i][attrib]
#                 self.attrib_data.append(attribs)

#         self.num_images = len(self.bbox_data)

#         # build sampling structure:
#         # each attribute has a dictionary pointing to all image indices which contain that attribute value
#         self.attrib_inds = []
#         for attrib in StreetStyleDataset.attributes:
#             self.attrib_inds.append({x : [] for x in range(0, len(attrib))})
#         # fill with all image attributes initially
#         for img_idx in range(0, self.num_images):
#             attribs = self.attrib_data[img_idx]
#             for attrib_idx, attrib_value in enumerate(attribs):
#                 if attrib_value != -1:
#                     self.attrib_inds[attrib_idx][attrib_value].append(img_idx)
                    
#         # create splits
# #         train_max = int(self.num_images*self.train_split_frac) + 1
# #         self.train_inds = range(0, train_max)
# #         eval_max = train_max + int(self.num_images*self.eval_split_frac) + 1
# #         self.eval_inds = range(train_max, eval_max)
# #         self.test_inds = range(eval_max, self.num_images)
        
#         num_eval = int(self.num_images*self.eval_split_frac)
#         num_test = int(self.num_images*self.test_split_frac)
#         self.eval_inds = []
#         self.test_inds = []
#         # perform stratified sampling to build eval/test sets so they actually contain all the attributes to test
#         # remove indices form the sampling structure as we use them for test/eval, then only training are left
#         for i in range(0, num_eval):
#             # sample a random value to use for current attribute
#             val = random.randint(0, len(StreetStyleDataset.attributes[self.cur_attrib])-1)
#             # get a random image index with that value
#             dict_list_idx = random.randint(0, len(self.attrib_inds[self.cur_attrib][val])-1)
#             img_idx = self.attrib_inds[self.cur_attrib][val][dict_list_idx]
#             # remove from structure
#             del self.attrib_inds[self.cur_attrib][val][dict_list_idx]
#             # update
#             self.cur_attrib = (self.cur_attrib + 1) % len(StreetStyleDataset.attributes)
#             # add to eval set
#             self.eval_inds.append(img_idx)
            
#         self.cur_attribute = 0
#         for i in range(0, num_test):
#             # sample a random value to use for current attribute
#             val = random.randint(0, len(StreetStyleDataset.attributes[self.cur_attrib])-1)
#             # get a random image index with that value
#             dict_list_idx = random.randint(0, len(self.attrib_inds[self.cur_attrib][val])-1)
#             img_idx = self.attrib_inds[self.cur_attrib][val][dict_list_idx]
#             # remove from structure
#             del self.attrib_inds[self.cur_attrib][val][dict_list_idx]
#             # update
#             self.cur_attrib = (self.cur_attrib + 1) % len(StreetStyleDataset.attributes)
#             # add to eval set
#             self.test_inds.append(img_idx)
#         self.cur_attribute = 0

#         # print(len(self.attrib_inds[9][1])) # num wearing hat
#         # print(len(self.attrib_inds[6][2])) # num wearing vneck

#     def next_train(self):
#         '''
#         Returns the next training mini-batch using stratified sampling.
#         '''
#         labels = np.zeros((self.batch_size, len(StreetStyleDataset.attributes)))
#         images = None
#         for i in range(0, self.batch_size):
#             # sample a random value to use for current attribute
#             val = random.randint(0, len(StreetStyleDataset.attributes[self.cur_attrib])-1)
#             # get a random image index with that value
#             img_idx = self.attrib_inds[self.cur_attrib][val][random.randint(0, len(self.attrib_inds[self.cur_attrib][val])-1)]
#             # update
#             self.cur_attrib = (self.cur_attrib + 1) % len(StreetStyleDataset.attributes)
#             # load the image
#             img = self.load_img(img_idx)
#             label = np.array(self.attrib_data[img_idx])
#             # add image and label to batch
#             if images is None:
#                 images = torch.Tensor(self.batch_size, img.shape[0], img.shape[1], img.shape[2])
#             images[i] = img
#             labels[i] = label
#         return images, torch.from_numpy(labels)

#     def next_eval(self):
#         '''
#         Returns the next mini-batch from the evaluation split or None if no
#         more eval data is available. If None, resets so next time called will get the
#         first eval batch and so on.
#         '''
#         if self.cur_eval_start_idx == len(self.eval_inds):
#             self.cur_eval_start_idx = 0
#             return None, None

#         images = None
#         # start at cur eval start idx
#         start_idx = self.cur_eval_start_idx
#         end_idx = self.cur_eval_start_idx + self.batch_size
#         # make sure doesn't go past end
#         end_idx = end_idx if end_idx <= len(self.eval_inds) else len(self.eval_inds)
#         self.cur_eval_start_idx = end_idx
#         img_indices = self.eval_inds[start_idx:end_idx]
#         labels = np.zeros((len(img_indices), len(StreetStyleDataset.attributes)))
#         for i, img_idx in enumerate(img_indices):
#             # load the image
#             img = self.load_img(img_idx)
#             label = np.array(self.attrib_data[img_idx])
#             if images is None:
#                 images = torch.Tensor(len(img_indices), img.shape[0], img.shape[1], img.shape[2])
#             images[i] = img
#             labels[i] = label
#         return images, torch.from_numpy(labels)

#     def next_test(self):
#         '''
#         Returns the next mini-batch from the test split or None if no
#         more test data is available. If None, resets so next time called will get the
#         first test batch and so on.
#         '''
#         if self.cur_test_start_idx == len(self.test_inds):
#             self.cur_test_start_idx = 0
#             return None, None

#         images = None
#         # start at cur eval start idx
#         start_idx = self.cur_test_start_idx
#         end_idx = self.cur_test_start_idx + self.batch_size
#         # make sure doesn't go past end
#         end_idx = end_idx if end_idx <= len(self.test_inds) else len(self.test_inds)
#         self.cur_test_start_idx = end_idx
#         img_indices = self.test_inds[start_idx:end_idx]
#         labels = np.zeros((len(img_indices), len(StreetStyleDataset.attributes)))
#         for i, img_idx in enumerate(img_indices):
#             # load the image
#             img = self.load_img(img_idx)
#             label = np.array(self.attrib_data[img_idx])
#             if images is None:
#                 images = torch.Tensor(len(img_indices), img.shape[0], img.shape[1], img.shape[2])
#             images[i] = img
#             labels[i] = label
#         return images, torch.from_numpy(labels)


#     def load_img(self, img_idx):
#         '''
#         Loads the given image with corresponding crop and transform applied.
#         '''
#         img_name = self.img_name_data[img_idx]
#         # load the image (located in path a/b/c/abcimagename.jpg)
#         img_path = os.path.join(self.image_data_dir, img_name[0], img_name[1], img_name[2], img_name)
#         img = Image.open(img_path)
#         # crop w/ bounding box
#         x1, y1, x2, y2 = self.bbox_data[img_idx]
#         img = img.crop((x1, y1, x2, y2))
#         if self.transform is not None:
#             img = self.transform(img)
#         return img

