import torch
import torch.nn as nn
from dataset import CustomDataset
from torch.utils.data import DataLoader

from model import generator, discriminator
from tqdm import tqdm
from torch import optim
from torchvision import utils, transforms
from torchvision.transforms import functional as trans_fn
import imageio


def save_gif(video, filename):
	x = video.permute(0,2,3,1)
	length = x.shape[0]
	x = x.numpy()
	frames = []
	for i in range(length):
		frames += [x[i]]
	imageio.mimsave(filename, frames)

def red(shape):
	y = torch.tensor([])
	for _ in range(3):
		x = torch.zeros(shape)
		x.unsqueeze_(0)
		y = torch.cat((y, x), 0)
	y[0] += 255/(255*1)
	y[1] += 153/(255*1)
	y[2] += 51/(255*1)
	print(f'y[1] = {y[1]}')
	return y

def green(shape):
	y = torch.tensor([])
	for _ in range(3):
		x = torch.zeros(shape)
		x.unsqueeze_(0)
		y = torch.cat((y, x), 0)
	y[0] += 19/(255*1)
	y[1] += 136/(255*1)
	y[2] += 8/(255*1)
	return y

def blue(shape):
	y = torch.tensor([])
	for _ in range(3):
		x = torch.zeros(shape)
		x.unsqueeze_(0)
		y = torch.cat((y, x), 0)
	y[2] += 128/(255*1)
	return y

if __name__ == '__main__':
	load_model = 1
	total_size = 60000
	batch_size = 128
	epoch = 200
	gen = nn.DataParallel(generator()).cuda()
	disc = nn.DataParallel(discriminator()).cuda()

	loss = nn.BCELoss()

	gen_optimizer = optim.Adam(gen.parameters(), lr = 0.0002, betas = (0.5, 0.999))
	disc_optimizer = optim.Adam(disc.parameters(), lr = 0.0002, betas = (0.5, 0.999))

	#my_data = CustomDataset()
	#print(f'batch_size = {batch_size}')
	#dataloader = DataLoader(my_data, batch_size = batch_size, shuffle = True)
	#print(f'len = {len(dataloader)}')
	if(load_model == 1):
		print(f'loaded model')
		checkpoint = torch.load('emnist_bceloss_model_200.model')
		gen.module.load_state_dict(checkpoint['gen'])
		disc.module.load_state_dict(checkpoint['disc'])
		gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
		disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
	
	red_back = red(shape = (21, 64))
	green_back = green(shape = (21, 64))
	blue_back = blue(shape = (22, 64))
	background = torch.cat((red_back, blue_back, green_back), 1)
	utils.save_image(
		background.unsqueeze(0),
		f'background.jpg')
	#background.unsqueeze_(0)
	with torch.no_grad():
		gen.eval()
		label = []
		name = '   happy    independence    day     '
		mapping = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10,'l':11,'m':12,'n':13,'o':14,'p':15,
		          'q':16,'r':17,'s':18,'t':19,'u':20,'v':21,'w':22,'x':23,'y':24,'z':25,' ':26}

		label = []
		gen_images = torch.tensor([])
		for c in name:
			#label.append(mapping[c])
			gen_inputs = torch.randn(1, 64)
			label = torch.LongTensor([mapping[c]]).cuda()
			cur_image = gen(input = gen_inputs, labels = label).data.cpu()
			print(f'cur_image shape = {cur_image.shape}')
			background = torch.cat((red_back, blue_back, green_back), 1)
			cur = background.unsqueeze(0)*2
			#print(f'cur = {cur.shape}')
			#print(f'cur_image = {cur_image.shape}')
			red_part = cur_image[:,:,:21,:]
			cur[:,:,:21,:] += red_part

			blue_part = cur_image[:,:,21:43,:]
			cur[:,:,21:43, :] += blue_part
			green_part = cur_image[:,:,43:,:]
			cur[:,:,43:, :] += green_part
			gen_images = torch.cat((gen_images, cur), 0)
			#gen_images = torch.cat((gen_images, cur_image),0)
		'''
		img_labels = torch.LongTensor(label).cuda()
		gen_inputs = torch.randn(len(label), 64).cuda()
		#print(f'img_labels')
		gen_images = gen(input = gen_inputs, labels = (img_labels)).data.cpu()
		#print(f' so')
		'''
		
		utils.save_image(
			gen_images,
			f'independence_day.jpg',
			nrow = 12,
			normalize = True,
			range = (-1, 1)
		)
		
		#save_gif(gen_images, 'happy.gif')