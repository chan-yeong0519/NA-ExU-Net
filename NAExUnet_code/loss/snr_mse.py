import torch.nn as nn

class LossFunction(nn.Module):
	def __init__(self, **kwargs):
		super(LossFunction, self).__init__()
		
		self.fc = nn.Linear(128, 1, bias=True)
		self.criterion  = nn.MSELoss()

		print('Initialised Mean Squared Error Loss')

	def forward(self, x, label=None):
        
		x = self.fc(x)
		nloss = self.criterion(x, label)
		return nloss