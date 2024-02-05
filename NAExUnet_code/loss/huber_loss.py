import torch.nn as nn
import torch

class LossFunction(nn.Module):
	def __init__(self, **kwargs):
		super(LossFunction, self).__init__()
		self.fc = nn.Linear(128, 1, bias=True)
		self.threshold = 1

		print('Initialised Huber Loss')

	def forward(self, x, label=None):
		x = self.fc(x)
		error = x - label
		is_small_error = torch.abs(error) <= self.threshold
		small_error_loss = torch.square(error) / 2
		big_error_loss = self.threshold * (torch.abs(error) - (0.5 * self.threshold))
		nloss = torch.where(is_small_error, small_error_loss, big_error_loss)
		nloss = torch.mean(nloss)
		return nloss