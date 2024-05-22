import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

# Define the Learner class, inheriting from nn.Module
class Learner(nn.Module):
	"""
	This is a learner class, which will accept a specific network module, such as OmniNet that define the network forward
	process. Learner class will create two same network, one as theta network and the other acts as theta_pi network.
	for each episode, the theta_pi network will copy its initial parameters from theta network and update several steps
	by meta-train set and then calculate its loss on meta-test set. All loss on meta-test set will be sumed together and
	then backprop on theta network, which should be done on metalaerner class.
	For learner class, it will be responsible for update for several steps on meta-train set and return with the loss on
	meta-test set.
	"""

	def __init__(self, net_cls, *args):
		"""
		It will receive a class: net_cls and its parameters: args for net_cls.
		:param net_cls: class, not instance
		:param args: the parameters for net_cls
		"""
		super(Learner, self).__init__()
		assert net_cls.__class__ == type

		# Create two instances of the network, one for theta and one for theta_pi
		self.net = net_cls(*args)
		self.net_pi = net_cls(*args)

		# Optimizer for updating theta_pi
		self.optimizer = optim.SGD(self.net_pi.parameters(), 0.1)

	def parameters(self):
		"""
		Override this function to return only net parameters for MetaLearner's optimize
		it will ignore theta_pi network parameters.
		:return:
		"""
		return self.net.parameters()

	def update_pi(self):
		"""
		copy parameters from self.net -> self.net_pi
		:return:
		"""
		for m_from, m_to in zip(self.net.modules(), self.net_pi.modules()):
			if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
				m_to.weight.data = m_from.weight.data.clone()
				if m_to.bias is not None:
					m_to.bias.data = m_from.bias.data.clone()

	def forward(self, support_x, support_y, query_x, query_y, num_updates):
		"""
		learn on current episode meta-train: support_x & support_y and then calculate loss on meta-test set: query_x&y
		:param support_x: [setsz, c_, h, w]
		:param support_y: [setsz]
		:param query_x:   [querysz, c_, h, w]
		:param query_y:   [querysz]
		:param num_updates: 5
		:return:
		"""
		# Copy theta_pi from theta network
		self.update_pi()

		# Update for several steps
		for i in range(num_updates):
			train_loss, _ = self.net_pi(support_x, support_y)
			self.optimizer.zero_grad()
			train_loss.backward()
			self.optimizer.step()

		# Compute the meta gradient and return it, the gradient is from one episode
		val_loss, predictions = self.net_pi(query_x, query_y)
		_, indices = torch.max(predictions, dim=1)  # Apply torch.max on predictions, not loss
		correct = torch.eq(indices, query_y).sum().item()
		acc = correct / query_y.size(0)

		# Compute gradients for validation on theta_pi
		grads_pi = autograd.grad(val_loss, self.net_pi.parameters(), create_graph=True)

		return train_loss, val_loss, grads_pi, acc

	def net_forward(self, support_x, support_y):
		"""
		This function is purely for updating net network. In metalearner, we need the get the loss op from net network
		to write our merged gradients into net network, hence will call this function to get a dummy loss op.
		:param support_x: [setsz, c, h, w]
		:param support_y: [sessz, c, h, w]
		:return: dummy loss and dummy pred
		"""
		loss, pred = self.net(support_x, support_y)
		return loss, pred


# Define the MetaLearner class, inheriting from nn.Module
class MetaLearner(nn.Module):
	"""
	As we have mentioned in Learner class, the metalearner class will receive a series of loss on different tasks/episodes
	on theta_pi network, and it will merage all loss and then sum over it. The summed loss will be backproped on theta
	network to update theta parameters, which is the initialization point we want to find.
	"""

	def __init__(self, net_cls, net_cls_args, n_way, k_shot, meta_batchsz, beta, num_updates):
		"""

		:param net_cls: class, not instance. the class of specific Network for learner
		:param net_cls_args: tuple, args for net_cls, like (n_way, imgsz)
		:param n_way:
		:param k_shot:
		:param meta_batchsz: number of tasks/episode
		:param beta: learning rate for meta-learner
		:param num_updates: number of updates for learner
		"""
		super(MetaLearner, self).__init__()

		self.n_way = n_way
		self.k_shot = k_shot
		self.meta_batchsz = meta_batchsz
		self.beta = beta
		self.num_updates = num_updates

		# Create a learner instance to learn on episodes and gather the loss together
		self.learner = Learner(net_cls, *net_cls_args)
		# Optimizer to update theta parameters, not theta_pi parameters
		self.optimizer = optim.Adam(self.learner.parameters(), lr=beta)

	def write_grads(self, dummy_loss, sum_grads_pi):
		"""
		write loss into learner.net, gradients come from sum_grads_pi.
		Since the gradients info is not calculated by general backward, we need this function to write the right gradients
		into theta network and update theta parameters as wished.
		:param dummy_loss: dummy loss, nothing but to write our gradients by hook
		:param sum_grads_pi: the summed gradients
		:return:
		"""

		# Register a hook on each parameter in the net that replaces the current dummy grad
		# with our grads accumulated across the meta-batch
		hooks = []

		for i, v in enumerate(self.learner.parameters()):
			def closure():
				ii = i
				return lambda grad: sum_grads_pi[ii]

			hooks.append(v.register_hook(closure()))

		# Use our summed gradients_pi to update the theta/net network
		self.optimizer.zero_grad()
		dummy_loss.backward()
		self.optimizer.step()

		# Remove the hooks to prevent memory explosion
		for h in hooks:
			h.remove()

	def forward(self, support_x, support_y, query_x, query_y):
		"""
		Here we receive a series of episode, each episode will be learned by learner and get a loss on parameters theta.
		we gather the loss and sum all the loss and then update theta network.
		setsz = n_way * k_shotf
		querysz = n_way * k_shot
		:param support_x: [meta_batchsz, setsz, c_, h, w]
		:param support_y: [meta_batchsz, setsz]
		:param query_x:   [meta_batchsz, querysz, c_, h, w]
		:param query_y:   [meta_batchsz, querysz]
		:return:
		"""
		sum_grads_pi = None
		meta_batchsz = support_y.size(0)

		accs = []
		total_train_loss = 0
		total_val_loss = 0

		# Iterate over each episode/task
		for i in range(meta_batchsz):
			train_loss, val_loss, grad_pi, episode_acc = self.learner(
				support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates
			)
			accs.append(episode_acc)
			total_train_loss += train_loss.item()
			total_val_loss += val_loss.item()
			if sum_grads_pi is None:
				sum_grads_pi = grad_pi
			else:
				sum_grads_pi = [torch.add(i, j) for i, j in zip(sum_grads_pi, grad_pi)]

		# As we already have the grads to update
		# We use a dummy forward/backward pass to get the correct grads into self.net
		# Use hook mechanism to write summed gradient into the network
		dummy_loss, _ = self.learner.net_forward(support_x[0], support_y[0])
		self.write_grads(dummy_loss, sum_grads_pi)

		avg_train_loss = total_train_loss / meta_batchsz
		avg_val_loss = total_val_loss / meta_batchsz

		return avg_train_loss, avg_val_loss, accs
	def pred(self, support_x, support_y, query_x, query_y):
		"""
		predict for query_x
		:param support_x:
		:param support_y:
		:param query_x:
		:param query_y:
		:return:
		"""
		meta_batchsz = support_y.size(0)

		accs = []
		# Iterate over each episode/task
		for i in range(meta_batchsz):
			_, _, _, episode_acc = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates)
			accs.append(episode_acc)

		return np.array(accs).mean()
