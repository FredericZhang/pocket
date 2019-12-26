"""
Useful transforms 

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torchvision

__all__ = [
    'to_tensor', 'ToTensor'
]

def _to_list_of_tensor(x, dtype=None):
    return [torch.as_tensor(item, dtype=dtype) for item in x]

def _to_tuple_of_tensor(x, dtype=None):
    return (torch.as_tensor(item, dtype=dtype) for item in x)

def _to_dict_of_tensor(x, dtype=None):
    return dict([(k, torch.as_tensor(v, dtype=dtype)) for k, v in x.items()])

def to_tensor(x, input_format='tensor', dtype=None):
    """Convert input data to tensor based on its format"""
    if input_format == 'tensor':
        return torch.as_tensor(x, dtype=dtype)
    elif input_format == 'pil':
        return torchvision.transforms.functional.to_tensor(x)
    elif input_format == 'list':
        return _to_list_of_tensor(x, dtype)
    elif input_format == 'tuple':
        return _to_tuple_of_tensor(x, dtype)
    elif input_format == 'dict':
        return _to_dict_of_tensor(x, dtype)
    else:
        raise ValueError("Unsupported format {}".format(input_format))

class ToTensor:
    """Convert to tensor"""
    def __init__(self, input_format='tensor', dtype=None):
        self.input_format = input_format
        self.dtype = dtype
    def __call__(self, x):
        return to_tensor(x, self.input_format, self.dtype)
    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += 'input_format=\'{}\''.format(self.input_format)
        reprstr += ', dtype='
        reprstr += repr(self.dtype)
        reprstr += ')'
        return reprstr
