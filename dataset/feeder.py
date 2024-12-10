import torch.utils.data as data
from copy import deepcopy
import numpy as np
from PIL import Image
import cv2

class TumorClassificationDataset(data.Dataset):
    def __init__(self, paths, ground_truths, transform=None):
        super().__init__()
        self.paths = np.array(paths)
        self.ground_truths = ground_truths
        self.transform = transform
        
        self.class_to_idx = {gt:i for i, gt in enumerate(np.unique(self.ground_truths))}
        self.idx_to_class = {v:k for k,v in self.class_to_idx.items()}
        self.targets = np.array([self.class_to_idx[gt] for gt in ground_truths])
        
    def __fetch_samples(self, data, targets, index):
        path = data[index]
        image = Image.open(path).convert('L')
        target = targets[index]
        target = target.astype(np.int64)    
        if self.transform:
            image = self.transform(image)  # Apply the transform
        return image, target
        
    def __getitem__(self, index):
        return self.__fetch_samples(self.paths, self.targets, index)
    
    def get_class_samples(self, class_name, index):
        class_ids = self.targets==self.class_to_idx[class_name]
        class_samples = self.paths[class_ids]
        class_targets =  self.targets[class_ids]
        index = index%len(class_samples)
        return self.__fetch_samples(class_samples, class_targets, index)
    
    def get_class_names(self):
        return list(self.class_to_idx.keys())
    
    def __len__(self):
        return len(self.paths)
    

class TumorSegmentationDataset(data.Dataset):
    def __init__(self, data, transform=None):
        super().__init__()
        self.data = data
        self.transform = transform
        
    def __fetch_samples(self, data, index):
        info_dict = data[index]
        path = info_dict['img_path']
        image = Image.open(path).convert('L')

        width, height = image.size
        segmentation = info_dict['ann_info']['segmentation']
        segmentation = np.array(segmentation[0], dtype=np.int32).reshape(-1, 2)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask = cv2.fillPoly(mask, [segmentation], color=1)
        mask = Image.fromarray(mask)
        
        if self.transform:
            image, mask = self.transform(image, mask)
            
        return image, mask
        
    def __getitem__(self, index):
        return self.__fetch_samples(self.data, index)
    
    def __len__(self):
        return len(self.data)
