from abc import ABC, abstractmethod

import torch.nn as nn


class BaseUpsampler(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, features, *args, **kwargs):
        pass
