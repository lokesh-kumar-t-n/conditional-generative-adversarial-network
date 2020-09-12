from emnist import extract_training_samples
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision import utils
import numpy as np
from model import generator
from model import discriminator

class CustomDataset(Dataset):
	def __init__(self):
		super(CustomDataset, self).__init__()

		self.trans = transforms.Compose([
			transforms.Resize(64),
			transforms.ToTensor(),
			transforms.Normalize(mean = [0.5], std = [0.5])
			]
		)
		self.letter_images = datasets.EMNIST('data_letters', 'letters', train=True, download = False, transform = self.trans)

	def __len__(self):
		return len(self.letter_images)

	def __getitem__(self, index):
		image = self.letter_images[index][0].transpose(1, 2)
		label = self.letter_images[index][1]
		
		return image, label
