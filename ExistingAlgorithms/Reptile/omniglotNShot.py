from omniglot import Omniglot
import torchvision.transforms as transforms
from PIL import Image
import os.path
import numpy as np

# Define the OmniglotNShot class for few-shot learning tasks
class OmniglotNShot():
	def __init__(self, root, batchsz, n_way, k_shot, k_query, imgsz):
		"""

		:param dataroot:
		:param batch_size:
		:param n_way:
		:param k_shot:
		"""

		self.resize = imgsz
		if not os.path.isfile(os.path.join(root, 'omni.npy')):
			# If omni.npy does not exist, download and process the dataset
			self.x = Omniglot(root, download=True,
			                  transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
			                                                transforms.Resize(self.resize),
			                                                lambda x: np.reshape(x, (self.resize, self.resize, 1))]))

			temp = dict()  # {label: [img1, img2, ..., img20], ...}
			for (img, label) in self.x:
				if label in temp:
					temp[label].append(img)
				else:
					temp[label] = [img]

			self.x = []
			for label, imgs in temp.items():  # Discard label info, each label contains 20 imgs
				self.x.append(np.array(imgs))

			# Convert to numpy array
			self.x = np.array(self.x)
			print('dataset shape:', self.x.shape)
			temp = []  # Free memory
			# Save all dataset into npy file
			np.save(os.path.join(root, 'omni.npy'), self.x)
		else:
			# If omni.npy exists, load it
			self.x = np.load(os.path.join(root, 'omni.npy'))

		self.x = self.x / 255
		np.random.shuffle(self.x)

		self.x_train, self.x_test = self.x[:1200], self.x[1200:]
		self.normalization()

		self.batchsz = batchsz
		self.n_cls = self.x.shape[0]
		self.n_way = n_way  # n way
		self.k_shot = k_shot  # k shot
		self.k_query = k_query  # k query

		# Save pointer of current read batch in total cache
		self.indexes = {"train": 0, "test": 0}
		self.datasets = {"train": self.x_train, "test": self.x_test}  # Original data cached
		print("train_shape", self.x_train.shape, "test_shape", self.x_test.shape)

		# Preload data cache for efficiency
		self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # Current epoch data cached
		                       "test": self.load_data_cache(self.datasets["test"])}

	def normalization(self):
		"""
		Normalizes our data, to have a mean of 0 and sdt of 1
		"""
		self.mean = np.mean(self.x_train)
		self.std = np.std(self.x_train)
		self.max = np.max(self.x_train)
		self.min = np.min(self.x_train)
		print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
		self.x_train = (self.x_train - self.mean) / self.std
		self.x_test = (self.x_test - self.mean) / self.std

		self.mean = np.mean(self.x_train)
		self.std = np.std(self.x_train)
		self.max = np.max(self.x_train)
		self.min = np.min(self.x_train)
		print("after norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)

	def load_data_cache(self, data_pack):
		"""
		Collects several batches data for N-shot learning
		:param data_pack: [cls_num, 20, 84, 84, 1]
		:return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
		"""
		#  take 5 way 1 shot as example: 5 * 1
		setsz = self.k_shot * self.n_way
		querysz = self.k_query * self.n_way
		data_cache = []

		# print('preload next 50 caches of batchsz of batch.')
		for sample in range(50):  # Number of episodes
			# Initialize arrays for support and query sets
			support_x = np.zeros((self.batchsz, setsz, self.resize, self.resize, 1))
			support_y = np.zeros((self.batchsz, setsz), dtype=np.int)
			query_x = np.zeros((self.batchsz, querysz, self.resize, self.resize, 1))
			query_y = np.zeros((self.batchsz, querysz), dtype=np.int)

			for i in range(self.batchsz):   # One batch means one set
				shuffle_idx = np.arange(self.n_way)
				np.random.shuffle(shuffle_idx)
				shuffle_idx_test = np.arange(self.n_way)
				np.random.shuffle(shuffle_idx_test)
				selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)

				for j, cur_class in enumerate(selected_cls):  # For each selected cls
					selected_imgs = np.random.choice(data_pack.shape[1], self.k_shot + self.k_query, False)

					# Meta-training: select the first k_shot images for each class as support images
					for offset, img in enumerate(selected_imgs[:self.k_shot]):
						support_x[i, shuffle_idx[j] * self.k_shot + offset, ...] = data_pack[cur_class][img]
						support_y[i, shuffle_idx[j] * self.k_shot + offset] = j   # Relative indexing

					# Meta-test: treat the following k_query images as query images
					for offset, img in enumerate(selected_imgs[self.k_shot:]):
						query_x[i, shuffle_idx_test[j] * self.k_query + offset, ...] = data_pack[cur_class][img]
						query_y[i, shuffle_idx_test[j] * self.k_query + offset] = j  # Relative indexing

			data_cache.append([support_x, support_y, query_x, query_y])
		return data_cache

	def __get_batch(self, mode):
		"""
		Gets next batch from the dataset with name.
		:param dataset_name: The name of the dataset (one of "train", "val", "test")
		:return:
		"""
		# Update cache if indexes exceed cached number
		if self.indexes[mode] >= len(self.datasets_cache[mode]):
			self.indexes[mode] = 0
			self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

		next_batch = self.datasets_cache[mode][self.indexes[mode]]
		self.indexes[mode] += 1

		return next_batch

	def get_batch(self, mode):

		"""
		Get next batch
		:return: Next batch
		"""
		x_support_set, y_support_set, x_target, y_target = self.__get_batch(mode)

		k = int(np.random.uniform(low=0, high=4))

		# Rotate all the support set images
		for i in np.arange(x_support_set.shape[0]):
			x_support_set[i, :, :, :, :] = self.__rotate_batch(x_support_set[i, :, :, :, :], k)

		# Rotate all the target set images
		for i in np.arange(x_target.shape[0]):
			x_target[i, :, :, :, :] = self.__rotate_batch(x_target[i, :, :, :, :], k)

		return x_support_set, y_support_set, x_target, y_target

	def __rotate_batch(self, batch_images, k):
		"""
		Rotates a whole image batch
		:param batch_images: A batch of images
		:param k: integer degree of rotation counter-clockwise
		:return: The rotated batch of images
		"""
		batch_size = len(batch_images)
		for i in np.arange(batch_size):
			batch_images[i] = np.rot90(batch_images[i], k)
		return batch_images


if __name__ == '__main__':
	# Example usage to visualize a set of images via tensorboard
	from torchvision.utils import make_grid
	from matplotlib import pyplot as plt
	from tensorboardX import SummaryWriter
	import time
	import torch

	plt.ion()

	# Initialize TensorBoard writer
	tb = SummaryWriter('runs', 'mini-imagenet')
	# Initialize the OmniglotNShot dataset
	db = OmniglotNShot('dataset', batchsz=20, n_way=5, k_shot=5, k_query=2)

	set_ = db.get_batch('train')
	while set_ != None:
		# Get the support and query sets
		support_x, support_y, query_x, query_y = set_
		print(support_y[0])
		print(query_y[0])

		# Convert numpy arrays to torch tensors and prepare for visualization
		support_x = torch.from_numpy(support_x).float().transpose(2, 4).repeat(1, 1, 3, 1, 1)
		query_x = torch.from_numpy(query_x).float().transpose(2, 4).repeat(1, 1, 3, 1, 1)
		support_y = torch.from_numpy(support_y).float()  # [batch, setsz, 1]
		query_y = torch.from_numpy(query_y).float()
		batchsz, setsz, c, h, w = support_x.size()

		support_x = make_grid(support_x[0], nrow=5)
		query_x = make_grid(query_x[0], nrow=2)

		# Visualize the support set images
		plt.figure('support x')
		plt.imshow(support_x.transpose(2, 0).transpose(1, 0).numpy())
		plt.pause(0.5)
		# Visualize the query set images
		plt.figure('query x')
		plt.imshow(query_x.transpose(2, 0).transpose(1, 0).numpy())
		plt.pause(0.5)

		# Log images to TensorBoard
		tb.add_image('support_x', support_x)
		tb.add_image('query_x', query_x)

		# Get the next batch
		set_ = db.get_batch('train')
		time.sleep(10)

	tb.close()
