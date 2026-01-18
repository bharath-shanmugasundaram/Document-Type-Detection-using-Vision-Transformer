import numpy as np
import pandas as pd 
import seaborn as sns 
import torch as t 
import torch.nn as nn 
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import dataloader,TensorDataset
from datasets import load_dataset



ds = load_dataset("chainyo/rvl-cdip")

