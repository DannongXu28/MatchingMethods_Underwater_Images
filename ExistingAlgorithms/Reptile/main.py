from omniglotNShot import OmniglotNShot
from meta import MetaLearner
from naive import Naive
from MiniImagenet import MiniImagenet

import torch
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


def main():
	# Hyperparameters for the meta-learning task
	meta_batchsz = 32
	n_way = 5
	k_shot = 1
	k_query = k_shot
	meta_lr = 1e-4
	num_updates = 5
	dataset = 'omniglot'

	# Load dataset based on the specified dataset type
	if dataset == 'omniglot':
		imgsz = 28
		db = OmniglotNShot('dataset', batchsz=meta_batchsz, n_way=n_way, k_shot=k_shot, k_query=k_query, imgsz=imgsz)

	elif dataset == 'mini-imagenet':
		imgsz = 84
		# Initialize the mini-imagenet dataset loaders
		mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=n_way, k_shot=k_shot, k_query=k_query,
		                    batchsz=10000, resize=imgsz)
		db = DataLoader(mini, meta_batchsz, shuffle=True, num_workers=4, pin_memory=True)
		mini_test = MiniImagenet('../mini-imagenet/', mode='test', n_way=n_way, k_shot=k_shot, k_query=k_query,
		                    batchsz=1000, resize=imgsz)
		db_test = DataLoader(mini_test, meta_batchsz, shuffle=True, num_workers=2, pin_memory=True)

	else:
		raise  NotImplementedError

	# Initialize the meta-learner model
	meta = MetaLearner(Naive, (n_way, imgsz), n_way=n_way, k_shot=k_shot, meta_batchsz=meta_batchsz, beta=meta_lr,
	                   num_updates=num_updates).cuda()

	# Initialize TensorBoard writer for logging
	tb = SummaryWriter('runs')


	# Main training loop
	for episode_num in range(1000):

		# Training phase
		if dataset == 'omniglot':
			# Get a batch of support and query sets
			support_x, support_y, query_x, query_y = db.get_batch('test')
			support_x = Variable( torch.from_numpy(support_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
			query_x = Variable( torch.from_numpy(query_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
			support_y = Variable(torch.from_numpy(support_y).long()).cuda()
			query_y = Variable(torch.from_numpy(query_y).long()).cuda()
		elif dataset == 'mini-imagenet':
			try:
				# Get the next batch from the DataLoader
				batch_test = iter(db).next()
			except StopIteration as err:
				# Reload the dataset if the DataLoader is exhausted
				mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=n_way, k_shot=k_shot, k_query=k_query,
				                    batchsz=10000, resize=imgsz)
				db = DataLoader(mini, meta_batchsz, shuffle=True, num_workers=4, pin_memory=True)

			support_x = Variable(batch_test[0]).cuda()
			support_y = Variable(batch_test[1]).cuda()
			query_x = Variable(batch_test[2]).cuda()
			query_y = Variable(batch_test[3]).cuda()

		# Perform a forward pass and backpropagation
		train_loss, val_loss, accs = meta(support_x, support_y, query_x, query_y)
		train_acc = np.array(accs).mean()

		# Log training metrics to TensorBoard every 20 episodes
		if episode_num % 20 == 0:
			tb.add_scalar('Accuracy/Training Accuracy', train_acc, episode_num)
			tb.add_scalar('Loss/Training Loss', train_loss, episode_num)
			tb.add_scalar('Loss/Validation Loss', val_loss, episode_num)

		# Testing phase
		if episode_num % 20 == 0:
			test_accs = []
			# Test the model multiple times to get an average accuracy
			for i in range(min(episode_num // 5000 + 3, 10)):
				if dataset == 'omniglot':
					support_x, support_y, query_x, query_y = db.get_batch('test')
					support_x = Variable( torch.from_numpy(support_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
					query_x = Variable( torch.from_numpy(query_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
					support_y = Variable(torch.from_numpy(support_y).long()).cuda()
					query_y = Variable(torch.from_numpy(query_y).long()).cuda()
				elif dataset == 'mini-imagenet':
					try:
						# Get the next batch from the test DataLoader
						batch_test = iter(db_test).next()
					except StopIteration as err:
						# Reload the test dataset if the DataLoader is exhausted
						mini_test = MiniImagenet('../mini-imagenet/', mode='test', n_way=n_way, k_shot=k_shot,
						                         k_query=k_query,
						                         batchsz=1000, resize=imgsz)
						db_test = DataLoader(mini_test, meta_batchsz, shuffle=True, num_workers=2, pin_memory=True)
					support_x = Variable(batch_test[0]).cuda()
					support_y = Variable(batch_test[1]).cuda()
					query_x = Variable(batch_test[2]).cuda()
					query_y = Variable(batch_test[3]).cuda()


				# Get the accuracy of the model on the query set
				test_acc = meta.pred(support_x, support_y, query_x, query_y)
				test_accs.append(test_acc)

			test_acc = np.array(test_accs).mean()
			#tb.add_scalar('Validation Accuracy', test_acc, episode_num)
			tb.add_scalar('Accuracy/Validation Accuracy', test_acc, episode_num)

			# Print training and test accuracies
			print('episode:', episode_num, '\tfinetune acc:%.6f' % train_acc, '\t\ttest acc:%.6f' % test_acc)
			#tb.add_scalar('test-acc', test_acc)
			#tb.add_scalar('finetune-acc', train_acc)
			#tb.add_scalar('test-acc', test_acc, episode_num)

if __name__ == '__main__':
	main()

