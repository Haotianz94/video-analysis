import numpy as np
import cv2
import sklearn
from sklearn.neighbors import NearestNeighbors
from scipy import stats

from newsAnchor_dataset_train import NewsAnchorDataset

class HairColorClassifier(object):
    PATCH_HEIGHT = 5
    PATCH_RATIO = 0.2

    def __init__(self):
        self.dataset = NewsAnchorDataset('../data/cloth/cloth_label/', '../data/newsAnchor_train_manifest_bbox.pkl', batch_size=1, 
                                        transform=None, test_split_size=15)
        
    def train(self):
        features = []
        labels = []
        image, label, bbox = self.dataset.next_train_in_order()
        while image is not None:
            hist_flat = self.calc_hair_hist(image, bbox)
            if label[14] != -1:
                # add to all features
                features.append(hist_flat)
                labels.append(label[14])
                
            image, label, bbox = self.dataset.next_train_in_order()
        self.nn = NearestNeighbors(n_neighbors=5)
        self.nn.fit(features)

        self.feature_data = np.array(features)
        self.label_data = np.array(labels)

    def infer(self, images, bboxes):
        '''
        Infers hair color from a batch of images of the same person.
        - images : [np.array, np.array, ...]
        - bboxes : np.array w/ shape (batch_size, 4)
        '''
        hist_flat = []
        for i in range(0, len(images)):
            hist_flat.append(self.calc_hair_hist(images[i], bboxes[i]))
        pred_label = self.nearest_neighbor_vote_predict(np.array(hist_flat), self.label_data, self.nn)
        return pred_label

    def calc_hair_hist(self, img, bbox):
        bbox_width = bbox[2] - bbox[0]
        patch_width = bbox_width * HairColorClassifier.PATCH_RATIO
        patch_offset = int(bbox_width / 2.0) - int(patch_width / 2.0)
        # create a mask
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[bbox[1]-HairColorClassifier.PATCH_HEIGHT:bbox[1], bbox[0]+patch_offset:bbox[2]-patch_offset] = 255
        # calculate histogram of hair
        hist = cv2.calcHist([img], [0, 1, 2], mask, [8, 8, 8], [0, 255, 0, 255, 0, 255])
        hist_flat = np.reshape(hist, (-1,))
        return np.ndarray.tolist(hist_flat)

    def nearest_neighbor_vote_predict(self, hist, labels, nn):
        dist, ind = nn.kneighbors(hist)
        neighbor_labels = labels[np.reshape(ind, (-1,))]
        print(neighbor_labels)
        mode, count = stats.mode(neighbor_labels, axis=None)
        print(mode[0])
        return mode[0]