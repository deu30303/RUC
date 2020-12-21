from PIL import Image
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import numpy as np
import pickle
import os


class CIFAR10RUC(datasets.CIFAR10):
    def __init__(self, root, transform, transform2, transform3, transform4=None, target_transform=None,train=True, download = False):
        self.root = root
        self.train = train  # training set or test set
        self.transform = transform
        self.transform2 = transform2
        self.transform3 = transform3
        self.transform4 = transform4

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

    def __getitem__(self, index) :
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform2(img)
            img3 = self.transform3(img)
            
        if self.transform4 != None:
            img4 = self.transform4(img)
            return img1, img2, img3, img4, target, index
        else:
            return img1, img2, img3, target, index

        return img1, img2, img3, target, index




class CIFAR20RUC(datasets.CIFAR100):
    def __init__(self, root, transform, transform2, transform3, transform4=None, target_transform=None,train=True, download = False):
        self.root = root
        self.train = train  # training set or test set
        self.transform = transform
        self.transform2 = transform2
        self.transform3 = transform3
        self.transform4 = transform4

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

    def __getitem__(self, index) :
        img, target = self.data[index], cifar100_to_cifar20(self.targets[index])
        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform2(img)
            img3 = self.transform3(img)
            
        if self.transform4 != None:
            img4 = self.transform4(img)
            return img1, img2, img3, img4, target, index
        else:
            return img1, img2, img3, target, index

        return img1, img2, img3, target, index
        

        
class STLRUC(datasets.STL10):
    def __init__(self, root, split='labeled', folds=None, transform=None, transform2=None, transform3=None, transform4 =None, target_transform=None, download=False):
        self.root = root
        self.split = split
        self.folds = folds
        self.transform = transform
        self.transform2 = transform2
        self.transform3 = transform3
        self.transform4 = transform4
        self.target_transform = target_transform

        if download:
            self.download()

        # now load the picked numpy arrays
        if self.split == 'train':
            self.data, self.labels = self.__loadfile(
                self.train_list[0][0], self.train_list[1][0])
            self.__load_folds(folds)

        elif self.split == 'train+unlabeled':
            self.data, self.labels = self.__loadfile(
                self.train_list[0][0], self.train_list[1][0])
            self.__load_folds(folds)
            unlabeled_data, _ = self.__loadfile(self.train_list[2][0])
            self.data = np.concatenate((self.data, unlabeled_data))
            self.labels = np.concatenate(
                (self.labels, np.asarray([-1] * unlabeled_data.shape[0])))

        elif self.split == 'unlabeled':
            self.data, _ = self.__loadfile(self.train_list[2][0])
            self.labels = np.asarray([-1] * self.data.shape[0])
            
        elif self.split == 'labeled':
            self.data, self.labels = self.__loadfile(self.train_list[0][0], self.train_list[1][0])
            self.__load_folds(folds)
            test_data, test_labels = self.__loadfile(self.test_list[0][0], self.test_list[1][0])
            self.data = np.concatenate((self.data, test_data))
            self.labels = np.concatenate((self.labels, test_labels))
        else:  # self.split == 'test':
            self.test_data, self.test_labels = self.__loadfile(self.test_list[0][0], self.test_list[1][0])
            
        class_file = os.path.join(self.root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()
                
    def __loadfile(self, data_file, labels_file=None):
        labels = None
        if labels_file:
            path_to_labels = os.path.join(
                self.root, self.base_folder, labels_file)
            with open(path_to_labels, 'rb') as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(self.root, self.base_folder, data_file)
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels
    
    def __load_folds(self, folds):
        # loads one of the folds if specified
        if folds is None:
            return
        path_to_folds = os.path.join(
            self.root, self.base_folder, self.folds_list_file)
        with open(path_to_folds, 'r') as f:
            str_idx = f.read().splitlines()[folds]
            list_idx = np.fromstring(str_idx, dtype=np.uint8, sep=' ')
            self.data, self.labels = self.data[list_idx, :, :, :], self.labels[list_idx]
                
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        if self.transform is not None:
            img1 = self.transform(img)
            if self.split == 'labeled' or 'train+unlabeled':
                img2 = self.transform2(img)
                img3 = self.transform3(img)
            if self.transform4 != None:
                img4 = self.transform4(img)
                
        if self.split == 'labeled' or 'train+unlabeled':
            if self.transform4 != None:
                return img1, img2, img3, img4, target, index
            else:
                return img1, img2, img3, target, index
        else:
            return img1, target, index
                
def cifar100_to_cifar20(target):
    """
    CIFAR100 to CIFAR 20 dictionary. 
    This function is from IIC github.
    """
  
    class_dict = {0: 4,
     1: 1,
     2: 14,
     3: 8,
     4: 0,
     5: 6,
     6: 7,
     7: 7,
     8: 18,
     9: 3,
     10: 3,
     11: 14,
     12: 9,
     13: 18,
     14: 7,
     15: 11,
     16: 3,
     17: 9,
     18: 7,
     19: 11,
     20: 6,
     21: 11,
     22: 5,
     23: 10,
     24: 7,
     25: 6,
     26: 13,
     27: 15,
     28: 3,
     29: 15,
     30: 0,
     31: 11,
     32: 1,
     33: 10,
     34: 12,
     35: 14,
     36: 16,
     37: 9,
     38: 11,
     39: 5,
     40: 5,
     41: 19,
     42: 8,
     43: 8,
     44: 15,
     45: 13,
     46: 14,
     47: 17,
     48: 18,
     49: 10,
     50: 16,
     51: 4,
     52: 17,
     53: 4,
     54: 2,
     55: 0,
     56: 17,
     57: 4,
     58: 18,
     59: 17,
     60: 10,
     61: 3,
     62: 2,
     63: 12,
     64: 12,
     65: 16,
     66: 12,
     67: 1,
     68: 9,
     69: 19,
     70: 2,
     71: 10,
     72: 0,
     73: 1,
     74: 16,
     75: 12,
     76: 9,
     77: 13,
     78: 15,
     79: 13,
     80: 16,
     81: 19,
     82: 2,
     83: 4,
     84: 6,
     85: 19,
     86: 5,
     87: 5,
     88: 8,
     89: 19,
     90: 18,
     91: 1,
     92: 2,
     93: 15,
     94: 6,
     95: 0,
     96: 17,
     97: 8,
     98: 14,
     99: 13}
    
    return class_dict[target]

