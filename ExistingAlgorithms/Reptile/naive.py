import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F

# Define the Naive network class, inheriting from nn.Module
class Naive(nn.Module):
	"""
	Define your network here.
	"""
	def __init__(self, n_way, imgsz):
		super(Naive, self).__init__()

		# Define the network architecture based on the image size
		if imgsz > 28: # For mini-imagenet
			self.net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3),
			                         nn.AvgPool2d(kernel_size=2),
			                         nn.BatchNorm2d(64),
			                         nn.ReLU(inplace=True),

			                         nn.Conv2d(64, 64, kernel_size=3),
			                         nn.AvgPool2d(kernel_size=2),
			                         nn.BatchNorm2d(64),
			                         nn.ReLU(inplace=True),

			                         nn.Conv2d(64, 64, kernel_size=3),
			                         nn.BatchNorm2d(64),
			                         nn.ReLU(inplace=True),

			                         nn.Conv2d(64, 64, kernel_size=3),
			                         nn.BatchNorm2d(64),
			                         nn.ReLU(inplace=True),

			                         nn.MaxPool2d(3,2)

			                         )
		else: # For omniglot
			self.net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3),
			                         nn.AvgPool2d(kernel_size=2),
			                         nn.BatchNorm2d(64),
			                         nn.ReLU(inplace=True),

			                         nn.Conv2d(64, 64, kernel_size=3),
			                         nn.AvgPool2d(kernel_size=2),
			                         nn.BatchNorm2d(64),
			                         nn.ReLU(inplace=True),

			                         nn.Conv2d(64, 64, kernel_size=3),
			                         nn.BatchNorm2d(64),
			                         nn.ReLU(inplace=True),

			                         nn.Conv2d(64, 64, kernel_size=3),
			                         nn.BatchNorm2d(64),
			                         nn.ReLU(inplace=True)

			                         )

		# Dummy forward to get the feature size after the convolutional layers
		dummy_img = Variable(torch.randn(2, 3, imgsz, imgsz))
		repsz = self.net(dummy_img).size()
		_, c, h, w = repsz
		self.fc_dim = c * h * w

		# Fully connected layers for classification
		self.fc = nn.Sequential(nn.Linear(self.fc_dim, 64),
		                        nn.ReLU(inplace=True),
		                        nn.Linear(64, n_way))

		# Loss function
		self.criteon = nn.CrossEntropyLoss()

		# Print the network architecture and representation size
		print(self)
		print('Naive repnet sz:', repsz)

	def forward(self, x, target):
		# Pass the input through the convolutional layers
		x = self.net(x)
		# Flatten the output of the convolutional layers
		x = x.view(-1, self.fc_dim)
		# Pass the flattened output through the fully connected layers
		pred = self.fc(x)
		# Compute the loss
		loss = self.criteon(pred, target)

		return loss, pred
