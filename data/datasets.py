from typing import OrderedDict
import torch
from torch.utils.data.dataset import random_split 
import torchvision
from torchvision import transforms
from torchvision import datasets as tvdatasets
import pytorch_lightning as pl 

import numpy as np 
import os 
from tqdm import tqdm 
from collections import Counter 
import random 
from sklearn.model_selection import train_test_split

from data import data, datasets, parser

class BasicDataset(torch.utils.data.Dataset):
    """ Basic Pytorch dataset, where data is held in 'x' and labels held in 'label'
    
    """
    def __init__(self, x, labels=None):
        """ Pass through data, labels and intermediate labels

        :param x: Data
        :type x: iterable
        :param labels: Labels. Defaults to None.
        :type labels: iterable, optional
        """
        self.x = x
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        """ Return dict of next item x and label 

        :param idx: Index
        :type idx: int

        :return: Dict with keys: x, label for data and label 
        :rtype: dict
        """
        item = {'x': self.x[idx]}
        if self.labels != None: item["label"] = self.labels[idx]
        return item['x'], item['label']

    def __len__(self) -> int:
        """ Get length of dataset

        :return: Length of dataset 
        :rtype: int
        """
        return len(self.x)

class BasicDatasetDict(torch.utils.data.Dataset):
    """ Basic Pytorch dataset, where data is held in 'x' and labels held in 'label'
    
    """
    def __init__(self, x, labels=None):
        """ Pass through data, labels and intermediate labels

        :param x: Data
        :type x: iterable
        :param labels: Labels. Defaults to None.
        :type labels: iterable, optional
        """
        self.x = x
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        """ Return dict of next item x and label 

        :param idx: Index
        :type idx: int

        :return: Dict with keys: x, label for data and label 
        :rtype: dict
        """
        item = {'x': self.x[idx]}
        if self.labels != None: item["label"] = self.labels[idx]
        return item

    def __len__(self) -> int:
        """ Get length of dataset

        :return: Length of dataset 
        :rtype: int
        """
        return len(self.x)

class EndEndDataset(torch.utils.data.Dataset):
    """ Dataset for end-to-end Neuroplytorch training, which has labels and intermediate labels, i.e. the labels for the perception model.

    """
    def __init__(self, x, labels=None, labels_intermediate=None):
        """ Pass through data, labels and intermediate labels

        :param x: Data
        :type x: iterable
        :param labels: Labels. Defaults to None.
        :type labels: iterable, optional
        :param labels_intermediate: Intermediate (perception model) labels. Defaults to None.
        :type labels_intermediate: iterable, optional
        """
        self.x = x
        self.labels = labels
        self.labels_intermediate = labels_intermediate

    def __getitem__(self, idx: int) -> dict:
        """ Return dict of next item x, label and intermediate label

        :param idx: Index
        :type idx: int

        :return: Dict with keys: x, label, label_intermediate for data, label and intermediate label
        :rtype: dict
        """
        item = {'x': self.x[idx]}
        if self.labels != None: item["label"] = self.labels[idx]
        if self.labels_intermediate != None: item["label_intermediate"] = self.labels_intermediate[idx]
        return item

    def __len__(self) -> int:
        """ Get length of dataset

        :return: Length of dataset 
        :rtype: int
        """
        return len(self.x)

class ReasoningDataModule(pl.LightningDataModule):
    """ Pytorch lightning data module for reasoning model training. Will create a synthetic dataset from given parameters to train the reasoning
    model.
    """
    def __init__(self, ce_fsm_list: list, ce_time_list: list, num_primitive_events: int, window_size: int, zero_windows: bool = False, 
                count_windows: bool = False, num_data: int = 1000000, batch_size: int=256, num_workers: int=int(os.cpu_count()/2),
                val_split: float=0.2, test_split: float=0.2):
        """ Init function for ReasoningDataModule

        :param ce_fsm_list: Pattern of primitive events for each complex event.
        :type ce_fsm_list: list
        :param ce_time_list: Temporal metadata pattern for each complex event
        :type ce_time_list: list
        :param num_primitive_events: Number of primitive events i.e. size of one hot primitive event vectors.
        :type num_primitive_events: int
        :param window_size: Size of the window of primitive events. 
        :type window_size: int
        :param zero_windows: Include windows that have no complex events if True. Defaults to False.
        :type zero_windows: bool, optional
        :param count_windows: Complex label is a count of complex events if True, otherwise a boolean vector where a complex event is 1 if there is at least one instance of that complex event. Defaults to False.
        :type count_windows: bool. optional
        :param num_data: Number of data items to generate. Defaults to 1000000.
        :type num_data: int, optional
        :param batch_size: Batch size of dataset loader. Defaults to 256.
        :type batch_size: int, optional
        :param num_workers: Number of CPU cores to load data. Defaults to int(os.cpu_count()/2).
        :type num_workers: int, optional
        :param val_split: Train-Validation split. Defaults to 0.2.
        :type val_split: float, optional
        :param test_split: Train/Validation-Test split. Defaults to 0.2.
        :type test_split: float, optional
        """
        super().__init__() 
        self.batch_size = batch_size 
        self.num_workers = num_workers 
        self.val_split = val_split 
        self.test_split = test_split

        self.ce_fsm_list = ce_fsm_list
        self.ce_time_list = ce_time_list
        self.num_primitive_events = num_primitive_events
        self.window_size = window_size
        self.zero_windows = zero_windows 
        self.count_windows = count_windows
        #self.balance_labels = balance_labels 
        self.num_data = num_data

        self.dl_dict = {'batch_size': self.batch_size, 'num_workers': self.num_workers, 'shuffle': True, 'persistent_workers': True}

    # TODO: balance labels or not?
    def prepare_data(self):
        """ Generate windowed primitive events and complex labels for reasoning model training
        """
        reasoning_windows, reasoning_labels, reasoning_labels_count = [], [], []

        # TODO: how to remove duplicates? - could throw the balance off potentially?
        for i in tqdm(range(self.num_data), total=self.num_data):
            w = data.generate_window(num_primitive_events=self.num_primitive_events, window_size=self.window_size)
            l = data.get_complex_label(window=w, ce_fsm_list=self.ce_fsm_list, ce_time_list=self.ce_time_list, count_windows=self.count_windows) 

            # if using boolean complex labels, make sure no element is >1
            if not self.count_windows and torch.max(l)>1: continue
            # if not using no-complex-event windows and window is labelled no-complex-event
            if not self.zero_windows and torch.sum(l)==0: continue

            reasoning_windows.append(w)
            reasoning_labels.append(l)
            reasoning_labels_count.append(tuple(l.tolist())) # used to count labels
            

        print(f'Number of primitive event windows {len(reasoning_labels)}')
        self.data_counter = Counter(reasoning_labels_count)
        print(f'Label balance', self.data_counter)
        
        self.reasoning_windows = reasoning_windows
        self.reasoning_labels = reasoning_labels

    def setup(self, stage: str=None):
        """ Preprocess data depending on the stage (fit or test)

        :param stage: String denoting the stage needed i.e. fit or test. Defaults to None, which is equivalent to fit.
        :type stage: str, optional 
        """
        train_test_split = int(len(self.reasoning_labels)*(1-self.test_split))
        self.fit_windows, self.fit_labels = self.reasoning_windows[:train_test_split], self.reasoning_labels[:train_test_split]
        self.test_windows, self.test_labels = self.reasoning_windows[train_test_split:], self.reasoning_labels[train_test_split:]

        if stage == 'fit' or stage is None:
            train_val_split = int(len(self.fit_windows)*(1-self.val_split)) 
            self.train_windows, self.val_windows = self.fit_windows[:train_val_split], self.fit_windows[train_val_split:]
            self.train_labels, self.val_labels = self.fit_labels[:train_val_split], self.fit_labels[train_val_split:]

            self.train_dataset = BasicDatasetDict(self.train_windows, self.train_labels) 
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, **self.dl_dict)
            self.val_dataset = BasicDatasetDict(self.val_windows, self.val_labels)
            self.val_loader = torch.utils.data.DataLoader(self.val_dataset, **self.dl_dict)

        if stage == 'test' or stage is None:
            self.test_dataset = BasicDatasetDict(self.test_windows, self.test_labels) 
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, **self.dl_dict)

        
    def train_dataloader(self): 
        """ Return Pytorch DataLoader for train dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.train_loader
    def val_dataloader(self): 
        """ Return Pytorch DataLoader for validation dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.val_loader 
    def test_dataloader(self): 
        """ Return Pytorch DataLoader for test dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.test_loader

class PerceptionWindowDataModule(pl.LightningDataModule):
    """ Dataset for training a perception model using windowed data. Useful for training a perception model similarly to end-to-end
    training but using the intermediate labels to calculate loss 
    """
    def __init__(self, dataset: dict, num_primitive_events: int, window_size: int, batch_size: int=256, num_workers: int=int(os.cpu_count()/2),
                val_split: float=0.2, test_split: float=0.2):
        """ 

        :param dataset: Tuple with keys 'train' and 'test', each of which holds a dataset (train and test set respectively). The dataset must be in a format that is iterable, where it can be iterated as data,label pairs. For example, can be a list of tuples, or a Pytorch dataset.
        :type dataset: tuple
        :param num_primitive_events: Number of primitive events i.e. size of one hot primitive event vectors.
        :type num_primitive_events: int
        :param window_size: Size of the window of primitive events.
        :type window_size: int
        :param batch_size: Batch size of dataset loader. Defaults to 256.
        :type batch_size: int 
        :param num_workers: Number of CPU cores to load data. Defaults to int(os.cpu_count()/2).
        :type num_workers: int, optional
        :param val_split: Train-Validation split. Defaults to 0.2.
        :type val_split: float, optional
        :param test_split: Train/Validation-Test split. Defaults to 0.2.
        :type test_split: float, optional
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size 
        self.num_workers = num_workers 
        self.val_split = val_split 
        self.test_split = test_split

        self.window_size = window_size
        self.num_primitive_events = num_primitive_events

        self.dl_dict = {'batch_size': self.batch_size, 'num_workers': self.num_workers, 'shuffle': True, 'persistent_workers': True}

    def prepare_data(self):
        """ Generate windowed raw input data for training a perception model using windowed data
        """
        print("Preparing dataset")
        # for train and test set 
        for t in ['train', 'test']:
            all_data = [] 
            
            # from Pytorch Dataset to list 
            for xx in self.dataset[t]: all_data.append(xx)
            random.shuffle(all_data) 

            # split data and labels
            datas, labels = [], []
            for xx in all_data: 
                datas.append(xx[0]) 
                labels.append(xx[1])
                print(xx[1], flush=True)

            # rolling window sampling with size window_size and stride=1
            image_windows, label_windows = [], [] 
            for i in tqdm(range(0, len(datas)-self.window_size), total=len(labels)-self.window_size):
                image_windows.append(torch.stack(datas[i:i+self.window_size]))
                label_windows.append(torch.stack([data.create_primitive_event(self.num_primitive_events,a) for a in labels[i:i+self.window_size]]))
            
            # if preparing train set
            if t:
                self.image_windows = torch.stack(image_windows).float()
                self.label_windows = torch.stack(label_windows).float() 
            else:
                self.image_windows_test = torch.stack(image_windows).float()
                self.label_windows_test = torch.stack(label_windows).float()

    def setup(self, stage=None):
        """ Preprocess data depending on the stage (fit or test)

        :param stage: String denoting the stage needed i.e. fit or test. Defaults to None, which is equivalent to fit.
        :type stage: str, optional 
        """
        if stage == 'fit' or stage is None:
            train_val_split = int(len(self.image_windows)*(1-self.val_split)) 
            self.train_windows, self.val_windows = self.image_windows[:train_val_split], self.image_windows[train_val_split:]
            self.train_labels, self.val_labels = self.label_windows[:train_val_split], self.label_windows[train_val_split:]

            self.train_dataset = BasicDatasetDict(self.train_windows, self.train_labels) 
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, **self.dl_dict)
            self.val_dataset = BasicDatasetDict(self.val_windows, self.val_labels)
            self.val_loader = torch.utils.data.DataLoader(self.val_dataset, **self.dl_dict)

        if stage == 'test' or stage is None:
            self.test_dataset = BasicDatasetDict(self.image_windows_test, self.label_windows_test) 
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, **self.dl_dict)

        
    def train_dataloader(self): 
        """ Return Pytorch DataLoader for train dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.train_loader
    def val_dataloader(self): 
        """ Return Pytorch DataLoader for validation dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.val_loader 
    def test_dataloader(self): 
        """ Return Pytorch DataLoader for test dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.test_loader
    
class EndToEndDataModule(pl.LightningDataModule):
    """ Pytorch Lightning data module to create the end-to-end training dataset for Neuroplytorch. Given a raw input dataset (e.g. MNIST) and the complex event parameters (patterns and metadata),
    this will preprocess the data to create the windowed data and necessary labels for end-to-end training. Especially useful in conjunction with the Pytorch Lightning Neuroplytorch model for
    easy and automated training. 
    """
    def __init__(self, dataset: tuple, ce_fsm_list: list, ce_time_list: list, num_primitive_events: int, window_size: int,
                zero_windows: dict={'train': True, 'val': True, 'test': True}, count_windows: dict={'train': False, 'val': False, 'test': False},
                batch_size: dict=256, num_workers: dict=int(os.cpu_count()/2), val_split: float=0.2):
        """ 

        :param dataset: Tuple with keys 'train' and 'test', each of which holds a dataset (train and test set respectively). The dataset must be in a format that is iterable, where it can be iterated as data,label pairs. For example, can be a list of tuples, or a Pytorch dataset.
        :type dataset: tuple
        :param ce_fsm_list: Pattern of primitive events for each complex event
        :type ce_fsm_list: list
        :param ce_time_list: Temporal metadata pattern for each complex event
        :type ce_time_list: list
        :param num_primitive_events: Number of primitive events i.e. size of one hot primitive event vectors. 
        :type num_primitive_events: int
        :param window_size: Size of the window of primitive events. 
        :type window_size: int
        :param zero_windows: Include windows that have no complex events if True. Dict of 'train', 'val' and 'test' to define this for each set, for example so the train set can exclude no complex event windows, but the validation set doesn't exclude. Defaults to {'train': True, 'val': True, 'test': True}.
        :type zero_windows: dict, optional
        :param count_windows: Complex label is a count of complex events if True. Dict of 'train', 'val' and 'test' to define this for each set. Defaults to {'train': False, 'val': False, 'test': False}.
        :type count_windows: dict, optional
        :param batch_size: Batch size of dataset loader. Defaults to 256.
        :type batch_size: int, optional
        :param num_workers: Number of CPU cores to load data. Defaults to int(os.cpu_count()/2).
        :type num_workers: int, optional
        :param val_split: Train-Validation split for the train dataset. Defaults to 0.2.
        :type val_split: float, optional
        """

        super().__init__()

        self.dataset = dataset 
        self.ce_fsm_list = ce_fsm_list 
        self.ce_time_list = ce_time_list
        self.num_primitive_events = num_primitive_events
        self.window_size = window_size
        
        self.zero_windows = zero_windows
        self.count_windows = count_windows
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        self.dl_dict = {'batch_size': self.batch_size, 'num_workers': self.num_workers, 'shuffle': True, 'persistent_workers': True}

    def prepare_data(self):
        """ Generate windowed raw input data with complex event labels for end to end training.
        """
        # for train and test set 
        for t in ['train', 'test']:
            print(f"Preparing {t} dataset")
            all_data = [] 

            # get data from iterable object (e.g list of tuples, Pytorch dataset etc.)
            for x in self.dataset[t]: all_data.append(x)
            random.shuffle(all_data) 

            datas, labels = [], []
            for xx in all_data: 
                datas.append(xx[0]) 
                labels.append(xx[1])

            # count the windows by complex event label, so show the bias of the (windowed) dataset - i.e. is there enough of each complex event to train?
            num_classes = len(self.ce_fsm_list)
            if self.zero_windows[t]: num_classes += 1
            class_counter = {} 
            data_windows, label_windows, complex_labels = [], [], [] 

            class_counter = {} 

            # sliding window sampling the data to create windowed data
            for i in tqdm(range(0, len(datas)-self.window_size), total=len(datas)-self.window_size):
                curr_window, curr_label = torch.stack(datas[i:i+self.window_size]), torch.stack([data.create_primitive_event(self.num_primitive_events,a) for a in labels[i:i+self.window_size]])
                curr_complex = data.get_complex_label(curr_label, self.ce_fsm_list, self.ce_time_list, self.count_windows[t])
                curr_simple = data.complex_to_simple(curr_complex)

                # if using boolean complex labels, make sure no element is >1
                if not self.count_windows and torch.max(curr_complex)>1: continue
                # if not using no-complex-event windows and window is labelled no-complex-event
                if not self.zero_windows[t] and torch.sum(curr_complex)==0: continue

                data_windows.append(curr_window)
                label_windows.append(curr_label)
                complex_labels.append(curr_complex)
                simple = data.complex_to_simple(curr_complex)
                class_counter[simple] = class_counter.get(simple, 0)+1
            
            # split train set into train-val sets 
            train_val_split = int(len(datas)*(1-self.val_split))
            if t=='train':
                self.data_windows = torch.stack(data_windows[:train_val_split]).float()
                self.label_windows = torch.stack(label_windows[:train_val_split]).float() 
                self.complex_labels = torch.stack(complex_labels[:train_val_split]).float()

                self.data_windows_val = torch.stack(data_windows[train_val_split:]).float() 
                self.label_windows_val = torch.stack(label_windows[train_val_split:]).float() 
                self.complex_labels_val = torch.stack(complex_labels[train_val_split:]).float() 
            else:
                self.data_windows_test = torch.stack(data_windows).float()
                self.label_windows_test = torch.stack(label_windows).float()
                self.complex_labels_test = torch.stack(complex_labels).float() 

            print(f"Class count for {t}: {class_counter}")

    def setup(self, stage=None):
        """ Preprocess data depending on the stage (fit or test)

        :param stage: String denoting the stage needed i.e. fit or test. Defaults to None, which is equivalent to fit.
        :type stage: str, optional 
        """
        if stage == 'fit' or stage is None: 
            self.train_dataset = EndEndDataset(self.data_windows, self.complex_labels, labels_intermediate=self.label_windows)
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, **self.dl_dict)

            self.val_dataset = EndEndDataset(self.data_windows_val, self.complex_labels_val, labels_intermediate=self.label_windows_val)
            self.val_loader = torch.utils.data.DataLoader(self.val_dataset, **self.dl_dict)

        if stage == 'test' or stage is None:
            #self.test_dataset = BasicDataset(self.image_windows_test, self.label_windows_test) 
            self.test_dataset = EndEndDataset(self.data_windows_test, self.complex_labels_test, labels_intermediate=self.label_windows_test)
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, **self.dl_dict)

        
    def train_dataloader(self): 
        """ Return Pytorch DataLoader for train dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.train_loader
    def val_dataloader(self): 
        """ Return Pytorch DataLoader for validation dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.val_loader 
    def test_dataloader(self): 
        """ Return Pytorch DataLoader for test dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.test_loader

class EndToEndNoTestDataModule(pl.LightningDataModule):
    """ Pytorch Lightning data module to create the end-to-end training dataset for Neuroplytorch. Given a raw input dataset (e.g. MNIST) and the complex event parameters (patterns and metadata),
    this will preprocess the data to create the windowed data and necessary labels for end-to-end training. Especially useful in conjunction with the Pytorch Lightning Neuroplytorch model for
    easy and automated training. Similar to EndToEndDataModule except uses the test set of the given dataset as the validation set, with no test set.
    """
    def __init__(self, dataset, ce_fsm_list: list, ce_time_list: list, num_primitive_events: int, window_size: int,
                zero_windows: dict={'train': True, 'val': True, 'test': True}, count_windows: dict={'train': False, 'val': False, 'test': False},
                batch_size: dict=256, num_workers: dict=int(os.cpu_count()/2), val_split: float=0.2):

        """ 

        :param dataset: Tuple with keys 'train' and 'test', each of which holds a dataset (train and test set respectively). The dataset must be in a format that is iterable, where it can be iterated as data,label pairs. For example, can be a list of tuples, or a Pytorch dataset.
        :type dataset: tuple
        :param ce_fsm_list: Pattern of primitive events for each complex event
        :type ce_fsm_list: list
        :param ce_time_list: Temporal metadata pattern for each complex event
        :type ce_time_list: list
        :param num_primitive_events: Number of primitive events i.e. size of one hot primitive event vectors. 
        :type num_primitive_events: int
        :param window_size: Size of the window of primitive events. 
        :type window_size: int
        :param zero_windows: Include windows that have no complex events if True. Dict of 'train', 'val' and 'test' to define this for each set, for example so the train set can exclude no complex event windows, but the validation set doesn't exclude. Defaults to {'train': True, 'val': True, 'test': True}.
        :type zero_windows: dict, optional
        :param count_windows: Complex label is a count of complex events if True. Dict of 'train', 'val' and 'test' to define this for each set. Defaults to {'train': False, 'val': False, 'test': False}.
        :type count_windows: dict, optional
        :param batch_size: Batch size of dataset loader. Defaults to 256.
        :type batch_size: int, optional
        :param num_workers: Number of CPU cores to load data. Defaults to int(os.cpu_count()/2).
        :type num_workers: int, optional
        """

        super().__init__()

        self.dataset = dataset 
        self.ce_fsm_list = ce_fsm_list 
        self.ce_time_list = ce_time_list
        self.num_primitive_events = num_primitive_events
        self.window_size = window_size

        self.zero_windows = zero_windows
        self.count_windows = count_windows
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
    
        self.dl_dict = {'batch_size': self.batch_size, 'num_workers': self.num_workers, 'shuffle': True, 'persistent_workers': True}

    def prepare_data(self):
        """ Generate windowed raw input data with complex event labels for end to end training.
        """
        # for train and test set 
        for t in ['train', 'test']:
            print(f"Preparing {t} dataset")
            all_data = [] 

            # get data from iterable object (e.g list of tuples, Pytorch dataset etc.)
            for x in self.dataset[t]: all_data.append(x)
            random.shuffle(all_data) 

            datas, labels = [], []
            for xx in all_data: 
                datas.append(xx[0]) 
                labels.append(xx[1])

            # count the windows by complex event label, so show the bias of the (windowed) dataset - i.e. is there enough of each complex event to train?
            num_classes = len(self.ce_fsm_list)
            if self.zero_windows[t]: num_classes += 1
            class_counter = {} 
            data_windows, label_windows, complex_labels = [], [], [] 

            class_counter = {} 
            class_no_zero_counter = {} 

            # sliding window sampling the data to create windowed data
            for i in tqdm(range(0, len(datas)-self.window_size), total=len(datas)-self.window_size):
                curr_window, curr_label = torch.stack(datas[i:i+self.window_size]), torch.stack([data.create_primitive_event(self.num_primitive_events,a) for a in labels[i:i+self.window_size]])
                curr_complex = data.get_complex_label(curr_label, self.ce_fsm_list, self.ce_time_list, self.count_windows[t])
                curr_simple = data.complex_to_simple(curr_complex)

                # if using boolean complex labels, make sure no element is >1
                if not self.count_windows and torch.max(curr_complex)>1: continue
                # if not using no-complex-event windows and window is labelled no-complex-event
                if not self.zero_windows[t] and torch.sum(curr_complex)==0: continue

                #MAX_0 = 10000
                #if True and t=='train': # if balance 
                #    if curr_simple==0 and class_counter.get(0,0)>=MAX_0: continue

                data_windows.append(curr_window)
                label_windows.append(curr_label)
                complex_labels.append(curr_complex)
                simple = data.complex_to_simple(curr_complex)
                class_counter[simple] = class_counter.get(simple, 0)+1 
                if simple: class_no_zero_counter[simple] = class_no_zero_counter.get(simple, 0)+1



            if t=='train':
                self.data_windows = torch.stack(data_windows).float()
                self.label_windows = torch.stack(label_windows).float() 
                self.complex_labels = torch.stack(complex_labels).float()
            else:
                self.data_windows_test = torch.stack(data_windows).float()
                self.label_windows_test = torch.stack(label_windows).float()
                self.complex_labels_test = torch.stack(complex_labels).float() 

            print(f"Class count for {t}: {class_counter}")

    def setup(self, stage=None):
        """ Preprocess data depending on the stage (fit or test)

        :param stage: String denoting the stage needed i.e. fit or test. Defaults to None, which is equivalent to fit.
        :type stage: str, optional 
        """
        if stage == 'fit' or stage is None: 
            self.train_dataset = EndEndDataset(self.data_windows, self.complex_labels, labels_intermediate=self.label_windows)
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, **self.dl_dict)

            self.test_dataset = EndEndDataset(self.data_windows_test, self.complex_labels_test, labels_intermediate=self.label_windows_test)
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, **self.dl_dict)

        

        
    def train_dataloader(self): 
        """ Return Pytorch DataLoader for train dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.train_loader
    def val_dataloader(self): 
        """ Return Pytorch DataLoader for validation dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.test_loader 

# CUSTOM DATA MODULES FOR INDIVIDUAL DATASETS
class MNISTDataModule(pl.LightningDataModule):
    """ Pytorch Lightning data module for MNIST dataset. Will load MNIST dataset from torchvision and cache locally in ./datasets/MNIST folder.
    Used for automatic data preprocessing, loading etc. for use with Pytorch Lightning module.

    """
    def __init__(self, data_dir="./datasets/MNIST", batch_size: int=256, num_workers=4, val_split=0.1):
        """
        :param data_dir: Directory to save local copy of dataset for offline caching. Defaults to "./datasets/MNIST".
        :type data_dir: str, optional
        :param batch_size: Batch size of dataset loader. Defaults to 256.
        :type batch_size: int, optional
        :param num_workers: Number of CPU cores to load data. Defaults to int(os.cpu_count()/2).
        :type num_workers: int, optional
        :param val_split: Train-Validation split for the train dataset. Defaults to 0.2.
        :type val_split: float, optional
        """
        super().__init__() 
        self.data_dir = data_dir 
        self.batch_size = batch_size 
        self.num_workers = num_workers 
        self.val_split = val_split

        self.transform = transforms.Compose([transforms.Resize((32, 32)),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])

        self.dl_dict = {'batch_size': self.batch_size, 'num_workers': self.num_workers, 'shuffle': True, 'persistent_workers': True}

    def prepare_data(self):
        """ Pytorch lightning overloaded method to prepare the data
        """
        # load train and test set for MNIST from torchvision, download if not cached 
        for t in [True, False]: tvdatasets.MNIST(self.data_dir, train=t, download=True)

    def setup(self, stage:str = None):
        """ Preprocess data depending on the stage (fit or test)

        Args:
            stage (str, optional): String denoting the stage needed i.e. fit or test. Defaults to None, which is equivalent to fit.
        """
        if stage == 'fit' or stage is None:
            mnist_full = tvdatasets.MNIST(self.data_dir, train=True, transform=self.transform)
            train_size = int(len(mnist_full)*(1-self.val_split))
            self.mnist_train, self.mnist_val = random_split(mnist_full, [train_size, len(mnist_full)-train_size]) 

            self.mnist_train_loader = torch.utils.data.DataLoader(self.mnist_train, **self.dl_dict)
            self.mnist_val_loader = torch.utils.data.DataLoader(self.mnist_val, **self.dl_dict)

        if stage == 'test' or stage is None:
            self.mnist_test = tvdatasets.MNIST(self.data_dir, train=False, transform=self.transform)
            self.mnist_test_loader = torch.utils.data.DataLoader(self.mnist_test, **self.dl_dict)

    def train_dataloader(self): 
        """ Return Pytorch DataLoader for train dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.mnist_train_loader
    def val_dataloader(self): 
        """ Return Pytorch DataLoader for validation dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.mnist_val_loader 
    def test_dataloader(self): 
        """ Return Pytorch DataLoader for test dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.mnist_test_loader

class EMNISTDataModule(pl.LightningDataModule):
    """ Pytorch Lightning data module for EMNIST dataset. Will load EMNIST dataset from torchvision and cache locally in ./datasets/EMNIST folder.
    Used for automatic data preprocessing, loading etc. for use with Pytorch Lightning module.

    """
    def __init__(self, data_dir="./datasets/EMNIST", batch_size: int=256, num_workers=4, val_split=0.1):
        """
        :param data_dir: Directory to save local copy of dataset for offline caching. Defaults to "./datasets/EMNIST".
        :type data_dir: str, optional
        :param batch_size: Batch size of dataset loader. Defaults to 256.
        :type batch_size: int, optional
        :param num_workers: Number of CPU cores to load data. Defaults to int(os.cpu_count()/2).
        :type num_workers: int, optional
        :param val_split: Train-Validation split for the train dataset. Defaults to 0.2.
        :type val_split: float, optional
        """
        super().__init__() 
        self.data_dir = data_dir 
        self.batch_size = batch_size 
        self.num_workers = num_workers 
        self.val_split = val_split

        self.transform = transforms.Compose([transforms.Resize((32, 32)),torchvision.transforms.ToTensor()])

        self.dl_dict = {'batch_size': self.batch_size, 'num_workers': self.num_workers, 'shuffle': True, 'persistent_workers': True}

    def prepare_data(self):
        """ Pytorch lightning overloaded method to prepare the data
        """
        # load train and test set for MNIST from torchvision, download if not cached 
        for t in [True, False]: tvdatasets.EMNIST(self.data_dir, train=t, download=True, split='letters')

    def setup(self, stage:str = None):
        """ Preprocess data depending on the stage (fit or test)

        :param stage: String denoting the stage needed i.e. fit or test. Defaults to None, which is equivalent to fit.
        :type stage: str, optional 
        """
        if stage == 'fit' or stage is None:
            emnist_full = tvdatasets.EMNIST(self.data_dir, train=True, transform=self.transform, split='letters')
            emnist_full.targets -= 1
            train_size = int(len(emnist_full)*(1-self.val_split))
            self.emnist_train, self.emnist_val = random_split(emnist_full, [train_size, len(emnist_full)-train_size]) 

            self.emnist_train_loader = torch.utils.data.DataLoader(self.emnist_train, **self.dl_dict)
            self.emnist_val_loader = torch.utils.data.DataLoader(self.emnist_val, **self.dl_dict)

        if stage == 'test' or stage is None:
            self.emnist_test = tvdatasets.EMNIST(self.data_dir, train=False, transform=self.transform, split='letters')
            self.emnist_test_loader = torch.utils.data.DataLoader(self.emnist_test, **self.dl_dict)

    def train_dataloader(self): 
        """ Return Pytorch DataLoader for train dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.emnist_train_loader
    def val_dataloader(self): 
        """ Return Pytorch DataLoader for validation dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.emnist_val_loader 
    def test_dataloader(self): 
        """ Return Pytorch DataLoader for test dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.emnist_test_loader

class UrbanDataModule(pl.LightningDataModule):
    """ 

    """
    def __init__(self, data_dir="./datasets/UrbanSound8K", batch_size: int=256, num_workers=4, val_split=0.1):
        """
        :param data_dir: Directory to save local copy of dataset for offline caching. Defaults to "./datasets/UrbanSound8K".
        :type data_dir: str, optional
        :param batch_size: Batch size of dataset loader. Defaults to 256.
        :type batch_size: int, optional
        :param num_workers: Number of CPU cores to load data. Defaults to int(os.cpu_count()/2).
        :type num_workers: int, optional
        :param val_split: Train-Validation split for the train dataset. Defaults to 0.2.
        :type val_split: float, optional
        """
        super().__init__() 
        self.data_dir = data_dir 
        self.batch_size = batch_size 
        self.num_workers = num_workers 
        self.val_split = val_split

        self.dl_dict = {'batch_size': self.batch_size, 'num_workers': self.num_workers, 'shuffle': True, 'persistent_workers': True}

    def prepare_data(self):
        """ Pytorch lightning overloaded method to prepare the data
        """
        # load train and test set for MNIST from torchvision, download if not cached 
        self.x = datasets.fetch_perception_data_local(dataset_loc=self.data_dir, dataset_type='audio_wav')

    def setup(self, stage:str = None):
        """ Preprocess data depending on the stage (fit or test)

        Args:
            stage (str, optional): String denoting the stage needed i.e. fit or test. Defaults to None, which is equivalent to fit.
        """
        if stage == 'fit' or stage is None:
            train_size = int(len(self.x['train'])*(1-self.val_split))
            self.audio_train, self.audio_val = random_split(self.x['train'], [train_size, len(self.x['train'])-train_size]) 

            self.audio_train_loader = torch.utils.data.DataLoader(self.audio_train, **self.dl_dict)
            self.audio_val_loader = torch.utils.data.DataLoader(self.audio_val, **self.dl_dict)

        if stage == 'test' or stage is None:
            self.audio_test = self.x['test']
            self.audio_test_loader = torch.utils.data.DataLoader(self.audio_test, **self.dl_dict)

    def train_dataloader(self): 
        """ Return Pytorch DataLoader for train dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.audio_train_loader
    def val_dataloader(self): 
        """ Return Pytorch DataLoader for validation dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.audio_val_loader 
    def test_dataloader(self): 
        """ Return Pytorch DataLoader for test dataset 

        :rtype: torch.utils.data.DataLoader
        """
        return self.audio_test_loader

def get_datamodule(datamodule_str: str) -> pl.LightningDataModule:
    """ Fetch the Pytorch Lightning DataModule specified by the string datamodule_str

    :param datamodule_str: Name of the Pytorch Lightning DataModule to return 
    :type datamodule_str: str

    :rtype: pytorch_lightning.LightningDataModule
    """
    if datamodule_str=='PerceptionWindowDataModule': return PerceptionWindowDataModule
    if datamodule_str=='ReasoningDataModule': return ReasoningDataModule
    if datamodule_str=='MNISTDataModule': return MNISTDataModule
    if datamodule_str=='EMNISTDataModule': return EMNISTDataModule
    if datamodule_str=='UrbanDataModule': return UrbanDataModule


def fetch_perception_data(dataset_str: str, dataset_loc: str='./datasets/MNIST') -> dict:
    """ Fetch raw input data from torchvision datasets

    :param dataset_str: Name of dataset from torchvision to fetch. Examples include MNIST, EMNIST etc.
    :type dataset_str: str
    :param dataset_loc: Directory to save a local copy of the dataset for offline caching. Defaults to ./datasets/MNIST
    :type dataset_loc: str, optional

    :return: Dictionary with keys for 'train' and 'test' split, each of which holds a torch.utils.data.Dataset dataset
    :rtype: dict
    """
    x = {} 
    if dataset_str=='MNIST':
        x['train'] = torchvision.datasets.MNIST(dataset_loc, train=True, download=True, transform=torchvision.transforms.Compose(
                [transforms.Resize((32, 32)),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))]
            ))
        x['test'] = torchvision.datasets.MNIST(dataset_loc, train=False, download=True, transform=torchvision.transforms.Compose(
                [transforms.Resize((32, 32)),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))]
            ))
    elif dataset_str=='EMNIST':
        x['train'] = torchvision.datasets.EMNIST('datasets/EMNIST', train=True, download=True, transform=torchvision.transforms.Compose(
                [transforms.Resize((32, 32)),torchvision.transforms.ToTensor()]), split='letters'
            )
        x['test'] = torchvision.datasets.EMNIST(dataset_loc, train=False, download=True, transform=torchvision.transforms.Compose(
                [transforms.Resize((32, 32)),torchvision.transforms.ToTensor()]), split='letters'
            )
        
        # EMNIST classes default 1-26, so need 0-25
        x['train'].targets -= 1
        x['test'].targets -= 1

        

    return x

def fetch_perception_data_local(dataset_loc: str, dataset_type: str, **kwargs) -> dict:
    """ Fetch raw input data from datasets folder

    :param dataset_loc: Location of dataset directory.
    :type dataset_loc: str
    :param dataset_type: Type of data in the dataset e.g. text, audio etc. 
    :type dataset_type: str

    :return: Dictionary with keys for 'train' and 'test' split, each of which holds a torch.utils.data.Dataset dataset
    :rtype: dict
    """
    x = {} 
    if dataset_type=='audio_wav':
        #xs, labels = parser.parse_waveforms(dataset_loc, **kwargs) 
        #xs_train, xs_test, ys_train, ys_test = train_test_split(xs, labels, test_size=0.2)
        #x['train'] = BasicDataset(xs_train, ys_train)
        #x['test'] = BasicDataset(xs_test, ys_test)

        xs_train, labels_train = parser.parse_waveforms(dataset_loc, split='train', **kwargs)
        xs_test, labels_test = parser.parse_waveforms(dataset_loc, split='test', **kwargs)

        x['train'] = BasicDataset(xs_train, labels=labels_train)
        x['test'] = BasicDataset(xs_test, labels=labels_test)


    return x
        
    










