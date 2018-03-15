from base_test import BaseTest
from newsAnchor_dataset_train import NewsAnchorDataset
from newsanchor_classifier_model import NewsAnchorClassifier

import numpy as np
import matplotlib.pyplot as plt
import time
import sys

import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms
from data_utils import ResizeTransform

class NewsAnchorClassifierTest(BaseTest):

    def __init__(self, streetstyle_model_path, use_gpu=True):
        super(self.__class__, self).__init__(use_gpu)
        self.log['train_acc'] = []
        self.log['val_acc'] = []
        #self.log['val_mean_class_acc'] = []
        #self.log['best_model_val_mean_class_acc'] = [0]*16
        self.log['best_model_val_acc'] = [0]*16
        self.log['best_model_val_loss'] = [float('inf')]*16

        pretrained_log = torch.load(streetstyle_model_path)
        self.pretrained_model = pretrained_log['best_model']


    def create_data_loaders(self):
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = NewsAnchorDataset('../data/cloth/cloth_label/', '../data/newsAnchor_train_manifest.pkl', batch_size=32, 
                                        transform=transform, test_split_size=15)
        # have to use custom loader b/c of sampling strategy
        # same class controls every split
        self.train_loader = dataset 
        self.test_loader = dataset

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
        images, labels = self.train_loader.next_train()
        img = torchvision.utils.make_grid(images[:16], nrow=4)
        self.imshow(img)

    def create_model(self):
        self.model = NewsAnchorClassifier()
        if self.use_gpu:
            self.model = self.model.cuda()
        # use the pretrained weights form streetstyle model
        self.copy_streetstyle_weights()
        # print(self.model.state_dict())

    def copy_streetstyle_weights(self):
        # modified version of pytorch source (http://pytorch.org/docs/master/_modules/torch/nn/modules/module.html#Module.load_state_dict)
        own_state = self.model.state_dict()
        for name, param in self.pretrained_model.state_dict().items():
            if name in own_state:
                try:
                    if own_state[name].size() == param.size():
                        # regular copy
                        own_state[name].copy_(param)
                    else:
                        # a layer we modified, copy into first n rows
                        own_state[name][:param.size()[0]].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))

    def create_loss_function(self):
        self.loss_function = nn.CrossEntropyLoss()

    def create_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2, weight_decay=0)

    def train_model(self, num_iters, **kwargs):
        visualize_batches = kwargs.get("visualize_every_n_batches", 20)
        save_batches = kwargs.get("save_every_n_batches", 50)
        
        log_file = './log/train_newsanchor_log_'+ str(int(time.time())) + '.txt'
        log_out = open(log_file, 'w')

        # counter for early stopping
        not_improved_for = 0
        early_stop = False

        running_loss = np.zeros((16)) # loss for each attribute
        running_correct = np.zeros((16)) # number correct classifications for each attribute
        running_total = np.zeros((16)) # total number of classifications attempted
        iter_count = 0
        for i in range(0, num_iters):
            # get next mini-batch
            images, labels = self.train_loader.next_train()
            if self.use_gpu:
                images = Variable(images.float().cuda(), requires_grad=False)
                labels = Variable(labels.long().cuda(), requires_grad=False)
            else:
                images = Variable(images.float(), requires_grad=False)
                labels = Variable(labels.long(), requires_grad=False)

            # zero param gradients
            self.optimizer.zero_grad()
            # forward pass
            output = self.model(images)
            # calculate loss and backward pass for each attribute
            # accumulate the gradients
            for j, attrib_output in enumerate(output):
                # only want loss for labeled attribs in image
                attrib_label = labels[:, j]
                good_indices = (attrib_label + 1).nonzero() # shift so nonlabeled attribs are 0 instead of -1
                good_indices = good_indices.view(good_indices.size(0))
                # remove unlabeled images in batch for this attrib
                attrib_label = attrib_label[good_indices]
                attrib_output = attrib_output[good_indices, :]
                # get loss
                attrib_loss = self.loss_function(attrib_output, attrib_label)
                # backward pass
                attrib_loss.backward(retain_graph=True)
                # save loss/accuracy
                running_loss[j] += attrib_loss.data[0]
                _, predicted = torch.max(attrib_output.data, 1)
                running_correct[j] += (predicted == attrib_label.data).sum()
                running_total[j] += attrib_label.size(0)
            iter_count += 1
            # optimize params
            self.optimizer.step()

            # save checkpoint if necessary
            # we log the current train/eval stats every time we save
            # and update best model if current model beats current best
            if i % save_batches == 0:
                # log training info
                running_loss /= iter_count
                running_correct = 1.*running_correct / running_total
                self.log['train_loss'].append(np.ndarray.tolist(running_loss))
                self.log['train_acc'].append(np.ndarray.tolist(running_correct))
                print('LOGGING MODEL AFTER %d Iters:' %(i))
                print('Training Loss: ' + str(running_loss))
                print('Training Accuracy: ' + str(running_correct))
                log_out.write('LOGGING MODEL AFTER %d Iters:\n' %(i))
                log_out.write('Training Loss: ' + str(running_loss) + '\n')
                log_out.write('Training Accuracy: ' + str(running_correct) + '\n')
                running_loss[:] = 0
                running_correct[:] = 0
                running_total[:] = 0
                iter_count = 0
                # run on evaluation set
#                 val_loss, val_acc, val_mean_class_acc = self.test_model()
                val_loss, val_acc = self.test_model()
                # log eval info
                self.log['val_loss'].append(np.ndarray.tolist(val_loss))
                self.log['val_acc'].append(np.ndarray.tolist(val_acc))
                #self.log['val_mean_class_acc'].append(np.ndarray.tolist(val_mean_class_acc))
                print('Eval Loss: ' + str(val_loss))
                print('Eval Accuracy: ' + str(val_acc))
                #print('Eval MCA: ' + str(val_mean_class_acc))
                log_out.write('Eval Loss: ' + str(val_loss) + '\n')
                log_out.write('Eval Accuracy: ' + str(val_acc) + '\n')
                #log_out.write('Eval MCA: ' + str(val_mean_class_acc) + '\n')
                # check if better than best model (according to mean class accuracy)
                best_model_loss_sum = np.sum(np.array(self.log['best_model_val_loss']))
                cur_model_loss_sum = np.sum(val_loss)
                if cur_model_loss_sum < best_model_loss_sum:
#                     self.log['best_model_val_mean_class_acc'] = np.ndarray.tolist(val_mean_class_acc)
                    self.log['best_model_val_acc'] = np.ndarray.tolist(val_acc)
                    self.log['best_model_val_loss'] = np.ndarray.tolist(val_loss)
                    self.log_best_model()
                    print('SAVED NEW BEST MODEL')
                    log_out.write('SAVED NEW BEST MODEL\n')
                    not_improved_for = 0
                else:
                    not_improved_for += 1
                    print('NOT IMPROVED FOR %d LOGS' %(not_improved_for))
                    log_out.write('NOT IMPROVED FOR %d LOGS\n' %(not_improved_for))
                    if (not_improved_for == 5):
                        print('EARLY STOPPING...')
                        log_out.write('EARLY STOPPING...\n')
                        early_stop = True
                # save log
                checkpoint = './log/newsanchor_'+ str(int(time.time())) + '_' + str(i) + '.tar'
                torch.save(self.log, checkpoint)
            elif i % visualize_batches == 0:
                # print update if necessary
                running_loss /= iter_count
                running_correct = 1.*running_correct / running_total
                print('After %d Iters:' %(i))
                print('Training Loss: ' + str(running_loss))
                print('Training Accuracy: ' + str(running_correct))
                log_out.write('After %d Iters\n:' %(i))
                log_out.write('Training Loss: ' + str(running_loss) + '\n')
                log_out.write('Training Accuracy: ' + str(running_correct) + '\n')
                running_loss[:] = 0
                running_correct[:] = 0
                running_total[:] = 0
                iter_count = 0
                
            sys.stdout.flush()
            log_out.flush()
            
            if early_stop:
                break
        
        print('FINISHED TRAINING!')
        log_out.write('FINISHED TRAINING!')
        log_out.close()

    def test_model(self):
        '''
        Evaluates the current model on the entire test split.
        Returns the loss, accuracy, and mean class accuracy.
        '''
        self.model.eval()
        # structures to keep track of per-class accuracy
        class_correct = []
        class_total = []
        for i in range(0, 16):
            num_attr_classes = len(NewsAnchorDataset.attributes[i])
            attr_count = np.zeros((num_attr_classes))
            attr_total = np.zeros((num_attr_classes))
            class_correct.append(attr_count)
            class_total.append(attr_total)

        running_loss = np.zeros((16)) # loss for each attribute
        running_correct = np.zeros((16)) # number correct classifications for each attribute
        running_total = np.zeros((16)) # total number of classifications attempted
        iter_count = 0
        # get first batch
        images, labels = self.test_loader.next_test()
        while images is not None:
            if self.use_gpu:
                images = Variable(images.float().cuda(), requires_grad=False)
                labels = Variable(labels.long().cuda(), requires_grad=False)
            else:
                images = Variable(images.float(), requires_grad=False)
                labels = Variable(labels.long(), requires_grad=False)

            # classify mini-batch
            output = self.model(images)
            # calculate loss/accuracy
            for j, attrib_output in enumerate(output):
                # only want loss for labeled attribs in image
                attrib_label = labels[:, j]
                good_indices = (attrib_label + 1).nonzero() # shift so nonlabeled attribs are 0 instead of -1
                good_indices = good_indices.view(good_indices.size(0))
                # remove unlabeled images in batch for this attrib
                attrib_label = attrib_label[good_indices]
                attrib_output = attrib_output[good_indices, :]
                # get loss
                attrib_loss = self.loss_function(attrib_output, attrib_label)
                # save loss/accuracy
                running_loss[j] += attrib_loss.data[0]
                _, predicted = torch.max(attrib_output, 1)
                running_correct[j] += (predicted.data == attrib_label.data).sum()
                running_total[j] += attrib_label.size(0)
                # save mean class accuracy info
                for k in range(0, len(class_correct[j])):
                    class_inds = (attrib_label == k).nonzero()
                    if len(class_inds) > 0:
                        class_inds = class_inds.view(class_inds.size(0))
                        predicted_class = predicted[class_inds]
                        class_correct[j][k] += (predicted_class == k).sum().data[0]
                        class_total[j][k] += len(predicted_class)
            iter_count += 1

            # next batch
            images, labels = self.test_loader.next_test()

        # calculate totals
        running_loss /= iter_count
        running_correct = 1.*running_correct / running_total
        #test_mean_class_acc = np.array([np.mean(1.*correct / total) for correct, total in zip(class_correct, class_total)])

        self.model.train()
        return running_loss, running_correct#, test_mean_class_acc
