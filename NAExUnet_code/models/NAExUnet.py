import torch
import torch.nn as nn
import torchaudio as ta
from models.ResNetBlocks import *

from torchsummary import summary as summary_
import arguments

class NAExU_Net(nn.Module): 
	def __init__(self, args):
		super(NAExU_Net, self).__init__()
		self.args = args
		self.l_channel = args['l_channel']
		self.encoder_split_level = args['encoder_split_level']
		self.ext_skip_conv_level = args['ext_skip_conv_level']
		self.ext_reduction_ratio = args['ext_reduction_ratio']
		self.l_num_convblocks = args['l_num_convblocks']
		self.code_dim = args['code_dim']
		self.stride = args['stride']
		self.first_kernel_size = args['first_kernel_size']
		self.first_stride_size = args['first_stride_size']
		self.first_padding_size = args['first_padding_size']

		self.inplanes   = self.l_channel[0]
		self.instancenorm   = nn.InstanceNorm1d(args['nfilts'])
		
		self.conv1 = nn.Conv2d(1, self.l_channel[0] , kernel_size=self.first_kernel_size, stride=self.first_stride_size, padding=self.first_padding_size,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(self.l_channel[0])
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(1, self.l_channel[0] , kernel_size=self.first_kernel_size, stride=self.first_stride_size, padding=self.first_padding_size,
							   bias=False)
		self.bn2 = nn.BatchNorm2d(self.l_channel[0])

		#################################################
		######### Description of Squeezing Path #########
		# level_1~N	: SE_ResBlock
		#################################################
		self.n_level = len(self.l_channel)
		
		# Encoder
		if not self.encoder_split_level == 0:
			for i in range(0, self.encoder_split_level):
				setattr(self, 'd_res_{}'.format(i+1), self._make_d_layer(SEBasicBlock, self.l_channel[i], self.l_num_convblocks[i], self.stride[i]))
		
		if not self.encoder_split_level >= self.n_level:
			for j in range(1, 3):
				for i in range(self.encoder_split_level, self.n_level):
					if i == 0:
						self.inplanes = self.l_channel[0]//2
					elif i == self.encoder_split_level:
						self.inplanes = self.l_channel[i-1]//2
					setattr(self, 'd_res_{}_{}'.format(j,i+1), self._make_d_layer(SEBasicBlock, self.l_channel[i]//2, self.l_num_convblocks[i], self.stride[i]))

		# Extractor
		self.inplanes =  self.l_channel[0]+self.l_channel[0]//self.ext_reduction_ratio[0]
		self.conv_channel = self.l_channel[0]
		for i in range(0, self.n_level):
			setattr(self, 'extractor_skip_con_conv_{}'.format(i+1), nn.Conv2d(self.conv_channel, self.conv_channel//self.ext_reduction_ratio[i], kernel_size=1, bias=False))
			setattr(self, 'd2_res_{}'.format(i+1), self._make_d_layer(SEBasicBlock, self.l_channel[i], self.l_num_convblocks[i], self.stride[i]))
			self.inplanes = self.l_channel[i]+self.l_channel[i]//self.ext_reduction_ratio[i+1]
			self.conv_channel = self.l_channel[i]

		#################################################
		######### Description of Expanding Path #########
		# level_0~N-1	: conv - BN
		# level_N		: conv
		#################################################
		for i in range(0, self.n_level):
			if i == 0: setattr(self, 'u_res_{}'.format(i), self._make_u_layer(SEBasicBlock, self.l_channel[-i-1], self.l_num_convblocks[-i-1]))
			else: setattr(self, 'u_res_{}'.format(i), self._make_u_layer(SEBasicBlock, self.l_channel[-i-1], self.l_num_convblocks[-i-1]))
			if i == 0: setattr(self, 'uconv_{}'.format(i), nn.Conv2d(self.l_channel[-i-1]*2, self.l_channel[-i-2], kernel_size=1, bias=False))
			elif i == self.n_level-1 : setattr(self, 'uconv_{}'.format(i), nn.Conv2d(self.l_channel[-i-1]*2, self.l_channel[0], kernel_size=1, bias=False))
			else: setattr(self, 'uconv_{}'.format(i), nn.ConvTranspose2d(self.l_channel[-i-1]*2, self.l_channel[-i-2], kernel_size=2, stride=2, bias=False))
		setattr(self, 'uconv_{}'.format(self.n_level), nn.ConvTranspose2d(self.l_channel[0]*2, 1, kernel_size=(2,1), stride = (2,1), bias=False))
			
		final_dim = self.l_channel[-1] * 8
		
		self.attention_encoder_sv = self._make_attention_module(final_dim//2, pooling=True)
		self.attention_encoder_noise = self._make_attention_module(final_dim//2, pooling=True)
		self.attention_encoder = self._make_attention_module(final_dim, pooling=True)
		self.attention_extractor = self._make_attention_module(final_dim, pooling=True)

		######################################################
		# speaker embedding layer and speaker identification #
		######################################################
		self.bn_agg_encoder = nn.BatchNorm1d(final_dim * 2)
		self.bn_agg_encoder_sv = nn.BatchNorm1d(final_dim)
		self.bn_agg_encoder_noise = nn.BatchNorm1d(final_dim)
		self.bn_agg_extractor = nn.BatchNorm1d(final_dim * 2)

		self.fc_sv_encoder = nn.Linear(final_dim, self.code_dim)
		self.fc_noise_encoder = nn.Linear(final_dim, self.code_dim)
		self.fc_extractor = nn.Linear(final_dim*4, self.code_dim)
		self.bn_sv_encoder_output = nn.BatchNorm1d(self.code_dim)
		self.bn_noise_encoder_output = nn.BatchNorm1d(self.code_dim)
		self.bn_code = nn.BatchNorm1d(self.code_dim)

		self.lRelu   = nn.LeakyReLU(negative_slope=0.2)
		self.relu    = nn.ReLU()
		self.drop    = nn.Dropout(0.5)
		
	def _make_d_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def _make_u_layer(self, block, planes, blocks, stride=1):
		downsample = None

		layers = []
		layers.append(block(planes, planes, stride, downsample))
		for i in range(1, blocks):
			layers.append(block(planes, planes))

		return nn.Sequential(*layers)

	def _make_attention_module(self, dim, pooling=False):
		if pooling == True:
			attention = nn.Sequential(
				nn.Conv1d(dim, dim//8, kernel_size=1),
				nn.ReLU(inplace=True),
				nn.BatchNorm1d(dim//8),
				nn.Conv1d(dim//8, dim, kernel_size=1),
				nn.Softmax(dim=-1),
			)
		else:
			attention = nn.Sequential(
				nn.Conv1d(dim, dim//8, kernel_size=1),
				nn.ReLU(inplace=True),
				nn.BatchNorm1d(dim//8),
				nn.Conv1d(dim//8, dim, kernel_size=1),
				nn.Sigmoid(),
			)
		return attention

	def forward(self, x, only_code=False):
		
		# spec augment
		if not only_code:
			freqm = ta.transforms.FrequencyMasking(self.args['spec_mask_F'])
			timem = ta.transforms.TimeMasking(self.args['spec_mask_T'])
	
			if self.args['spec_mask_F'] != 0:
				x = freqm(x)			
			if self.args['spec_mask_T'] != 0:
				x = timem(x)

		x = self.instancenorm(x).unsqueeze(1).detach()

		d_dx = {}
		d_sv_dx = {}
		d_noise_dx = {}
		u_dx = {}
		uc_sv_dx = {}
		uc_noise_dx = {}
		uc_dx = {}

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		d_dx['0'] = x

		# +++++++++++++++++++ Squeezing Path +++++++++++++++++++++ #
		if not self.encoder_split_level == 0:
			for i in range(1, self.encoder_split_level+1):
				d_dx['%d'%(i)] = getattr(self, 'd_res_{}'.format(i))(d_dx['%d'%(i-1)])
		
		c = d_dx['%d'%(self.encoder_split_level)].size(1)
		if not self.encoder_split_level == self.n_level:
			for j in range(1,3):
				if j == 1:
					for i in range(self.encoder_split_level+1, self.n_level+1):
						if i == self.encoder_split_level+1:
							d_sv_dx['%d'%(i-1)] = d_dx['%d'%(i-1)][:,:c//2,:,:]
						d_sv_dx['%d'%i] = getattr(self, 'd_res_{}_{}'.format(j,i))(d_sv_dx['%d'%(i-1)])
				elif j == 2:
					for i in range(self.encoder_split_level+1, self.n_level+1):
						if i == self.encoder_split_level+1:
							d_noise_dx['%d'%(i-1)] = d_dx['%d'%(i-1)][:,c//2:,:,:]
						d_noise_dx['%d'%i] = getattr(self, 'd_res_{}_{}'.format(j,i))(d_noise_dx['%d'%(i-1)])

		elif self.encoder_split_level == self.n_level:
			d_sv_dx['%d'%(self.n_level)] = d_dx['%d'%(self.n_level)][:,:c//2,:,:]
			d_noise_dx['%d'%(self.n_level)] = d_dx['%d'%(self.n_level)][:,c//2:,:,:]

		bs, c, _, time = d_sv_dx['%d'%(self.n_level)].size()
		encoder_sv_feature = d_sv_dx['%d'%(self.n_level)]
		encoder_noise_feature = d_noise_dx['%d'%(self.n_level)]

		sv_x = encoder_sv_feature.reshape(bs, -1, time)
		noise_x = encoder_noise_feature.reshape(bs, -1, time)

		w = self.attention_encoder_sv(sv_x)
		m = torch.sum(sv_x * w, dim=-1)
		s = torch.sqrt((torch.sum((sv_x ** 2) * w, dim=-1) - m ** 2).clamp(min=1e-5))
		sv_x = torch.cat([m, s], dim=1)
		encoder_sv_x = self.bn_agg_encoder_sv(sv_x)

		w = self.attention_encoder_noise(noise_x)
		m = torch.sum(noise_x * w, dim=-1)
		s = torch.sqrt((torch.sum((noise_x ** 2) * w, dim=-1) - m ** 2).clamp(min=1e-5))
		noise_x = torch.cat([m, s], dim=1)
		encoder_noise_x = self.bn_agg_encoder_noise(noise_x)

		encoder_sv_output = self.fc_sv_encoder(encoder_sv_x)
		encoder_sv_output = self.bn_sv_encoder_output(encoder_sv_output)

		encoder_noise_output = self.fc_noise_encoder(encoder_noise_x[bs//2:,:])
		encoder_noise_output = self.bn_noise_encoder_output(encoder_noise_output)

		encoder_x = torch.cat([encoder_sv_x, encoder_noise_x], dim=1)

		# +++++++++++++++++++ Expanding Path +++++++++++++++++++++ #
		u_sv_input = d_sv_dx['%d'%(self.n_level)]
		u_noise_input = d_noise_dx['%d'%(self.n_level)]
		u_input = torch.cat((u_sv_input,u_noise_input), 1)

		for i in range(0, self.n_level):
			u_dx['%d'%(i)] = getattr(self, 'u_res_{}'.format(i))((u_input))
			if self.n_level - i > self.encoder_split_level:
				d_dx['%d'%(self.n_level-i)] = torch.cat((d_sv_dx['%d'%(self.n_level-i)],d_noise_dx['%d'%(self.n_level-i)]),1)
				u_input = torch.cat((u_dx['%d'%(i)],d_dx['%d'%(self.n_level-i)]), 1)
			else:
				u_input = torch.cat((u_dx['%d'%(i)],(d_dx['%d'%(self.n_level-i)])), 1)

			u_input = getattr(self, 'uconv_{}'.format(i))((u_input))
			uc_dx['%d'%(i)] = u_input

		output = torch.cat((u_input, d_dx['0']), 1)
		output = getattr(self, 'uconv_{}'.format(self.n_level))((output))

		# +++++++++++++++++++ Requeezing Path +++++++++++++++++++++ #

		# spec augment
		if not only_code:
			freqm2 = ta.transforms.FrequencyMasking(self.args['spec_mask_F'])
			timem2 = ta.transforms.TimeMasking(self.args['spec_mask_T'])
	
			if self.args['spec_mask_F'] != 0:
				output = freqm2(output)			
			if self.args['spec_mask_T'] != 0:
				output = timem2(output)

		x = self.conv2(output)
		x = self.bn2(x)
		x = self.relu(x)

		for i in range(1, self.n_level+1):
			if i <= self.ext_skip_conv_level:
				uc_x = uc_dx['%d'%(self.n_level-i)]
			else:
				uc_x = getattr(self, 'extractor_skip_con_conv_{}'.format(i))((uc_dx['%d'%(self.n_level-i)]))
			x = torch.cat((x, uc_x), 1)
			x = getattr(self, 'd2_res_{}'.format(i))(x)
		
		x = x.reshape(bs, -1, time)

		w = self.attention_extractor(x)
		m = torch.sum(x * w, dim=-1)
		s = torch.sqrt((torch.sum((x ** 2) * w, dim=-1) - m ** 2).clamp(min=1e-5))
		x = torch.cat([m, s], dim=1)
		extractor_x = self.bn_agg_extractor(x)

		x = torch.cat([encoder_x, extractor_x], dim=1)

		code = self.fc_extractor(x)
		code = self.bn_code(code)

		if only_code: return code

		return code, output, encoder_sv_output, encoder_noise_output

if __name__ == '__main__':
	args, system_args, experiment_args = arguments.get_args()
	model = NAExU_Net(args).cpu()
	summary_(model, (64,256), batch_size=120, device='cpu')