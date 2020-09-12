import torch
import torch.nn as nn

class generator(nn.Module):
	def __init__(self):
		super(generator, self).__init__()
		self.embed = nn.Embedding(26, 16)#
		'''
		nn.Embedding(num_of_categories, size of the tensor the particular category should be embedded into)

		embedding is like a look up table with 0-indexing. the embedding are randomly initialized and these embedding are trained.
		The similar looking categories have similar embedding. As against to using one-hot encoding which does not have this property
		'''
		self.linear = nn.ModuleList(
			[
			 nn.Sequential(nn.Linear(64 + 16, 64 * 32), nn.ReLU()),
			 nn.Sequential(nn.Linear(64 * 32, 64 * 32), nn.ReLU()),
			 nn.Sequential(nn.Linear(64 * 32, 64 * 64), nn.Tanh())
			]
		)


	def forward(self, input, labels):
		embed = self.embed(labels)
		out = torch.cat((input, embed), -1)
		for i in range(len(self.linear)):
			out = self.linear[i](out)

		out = out.reshape(-1, 1, 64, 64)
		return out

class discriminator(nn.Module):
	def __init__(self):
		super(discriminator, self).__init__()
		self.embed = nn.Embedding(26, 64)

		self.linear = nn.ModuleList(
			[
			 nn.Sequential(nn.Linear(64 * 64 + 64, 64 * 32), nn.LeakyReLU(0.2)),
			 nn.Sequential(nn.Linear(64*32, 64*32), nn.LeakyReLU(0.2)),
			 nn.Sequential(nn.Linear(64*32, 1),nn.Sigmoid())
			]
		)

	def forward(self, input, labels):
		embed = self.embed(labels)
		out = input
		out = out.reshape(-1, 64*64)
		out = torch.cat((out, embed), -1)
		for i in range(len(self.linear)):
			out = self.linear[i](out)
		
		return out