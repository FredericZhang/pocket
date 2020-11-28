"""
Dataset base classes

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import pickle

from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Tuple

__all__ = ['DataDict', 'ImageDataset', 'DataSubset']

class DataDict(dict):
    r"""
    Data dictionary class. This is a class based on python dict, with
    augmented utility for loading and saving
    
    Arguments:
        input_dict(dict, optional): A Python dictionary
        kwargs: Keyworded arguments to be stored in the dict

    Example:

        >>> from pocket.data import DataDict
        >>> person = DataDict()
        >>> person.is_empty()
        True
        >>> person.age = 15
        >>> person.sex = 'male'
        >>> person.save('./person.pkl', 'w')
    """
    def __init__(self, input_dict=None, **kwargs):
        data_dict = dict() if input_dict is None else input_dict
        data_dict.update(kwargs)
        super(DataDict, self).__init__(**data_dict)

    def __getattr__(self, name):
        """Get attribute"""
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        """Set attribute"""
        self[name] = value

    def save(self, path, mode='wb', **kwargs):
        """Save the dict into a pickle file"""
        with open(path, mode) as f:
            pickle.dump(self.copy(), f, **kwargs)

    def load(self, path, mode='rb', **kwargs):
        """Load a dict or DataDict from pickle file"""
        with open(path, mode) as f:
            data_dict = pickle.load(f, **kwargs)
        for name in data_dict:
            self[name] = data_dict[name]

    def is_empty(self):
        return not bool(len(self))

class StandardTransform:
    """https://github.com/pytorch/vision/blob/master/torchvision/datasets/vision.py"""

    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, inputs: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)

class ImageDataset(Dataset):
    """
    Base class for image dataset

    Arguments:
        root(str): Root directory where images are downloaded to
        transform(callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version
        target_transform(callable, optional): A function/transform that takes in the
            target and transforms it
        transforms (callable, optional): A function/transform that takes input sample 
            and its target as entry and returns a transformed version.
    """
    def __init__(self, root, transform=None, target_transform=None, transforms=None):
        self._root = root
        self._transform = transform
        self._target_transform = target_transform
        if transforms is None:
            self._transforms = StandardTransform(transform, target_transform)
        elif transform is not None or target_transform is not None:
            print("WARNING: Argument transforms is given, transform/target_transform are ignored.")
            self._transforms = transforms
        else:
            self._transforms = transforms

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError
    
    def __repr__(self):
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=\"' + repr(self._root)
        reprstr += '\")'
        # Ignore the optional arguments
        return reprstr

    def __str__(self):
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tRoot path: {}\n'.format(self._root)
        return reprstr

    def load_image(self, path):
        """Load an image and apply transform"""
        return Image.open(path)

class DataSubset(Dataset):
    """
    A subset of data with access to all attributes of original dataset

    Arguments:
        dataset(Dataset): Original dataset
        pool(List[int]): The pool of indices for the subset
    """
    def __init__(self, dataset, pool):
        self.dataset = dataset
        self.pool = pool
    def __len__(self):
        return len(self.pool)
    def __getitem__(self, idx):
        return self.dataset[self.pool[idx]]
    def __getattr__(self, key):
        if hasattr(self.dataset, key):
            return getattr(self.dataset, key)
        else:
            raise AttributeError("Given dataset has no attribute \'{}\'".format(key))