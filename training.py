import torch
import torch.nn as nn
from dataset import CustomDataset
from torch.utils.data import DataLoader

from model import generator, discriminator
from tqdm import tqdm
from torch import optim
from torchvision import utils, transforms
from torchvision.transforms import functional as trans_fn

def training(epoch, gen, disc, gen_optimizer, disc_optimizer, loss, dataloader):
	image_set = iter(dataloader)
	print_img_every = 1
	pbar = tqdm(range(epoch))
	disc_weight = 1
	gen_weight = 1
	save_model_every = 50
	for i in pbar:
		image_set = iter(dataloader)
		while (True):
			try:
				real_images, cgan_labels = next(image_set)
			except (OSError, StopIteration):
				image_set = iter(dataloader)
				real_images, cgan_labels = next(image_set)
			
			real_images = real_images.cuda()
			disc_optimizer.zero_grad()
			gen.eval()
			disc.train()

			real_labels = torch.ones(real_images.shape[0]).cuda()
			real_predict = disc(input = real_images, labels = (cgan_labels - 1))
			real_predict = real_predict.squeeze()
			real_disc_loss = loss(real_predict, real_labels)
			real_disc_loss = real_disc_loss * disc_weight
			real_disc_loss.backward()

			gen_inputs = torch.randn(real_images.shape[0], 64).cuda()
			gen_images = gen(input = gen_inputs, labels = (cgan_labels - 1))

			fake_labels = torch.zeros(real_images.shape[0]).cuda()
			fake_predict = disc(input = gen_images, labels = (cgan_labels - 1))
			fake_predict = fake_predict.squeeze()
			fake_disc_loss = loss(fake_predict, fake_labels)
			
			fake_disc_loss = fake_disc_loss * disc_weight
			fake_disc_loss.backward()
			
			disc_loss = real_disc_loss + fake_disc_loss
			
			disc_optimizer.step()

			gen_optimizer.zero_grad()
			gen.train()
			disc.eval()

			gen_inputs = torch.randn(real_images.shape[0], 64).cuda()
			gen_images = gen(input = gen_inputs, labels = (cgan_labels - 1))

			
			gen_predict = disc(input = gen_images, labels = (cgan_labels - 1))
			gen_predict = gen_predict.squeeze()
			gen_loss = loss(gen_predict, real_labels)
			gen_loss = gen_loss * gen_weight
			gen_loss.backward()

			gen_optimizer.step()
			if((i + 1) % 1 == 0):
				msg = f'gen_loss : {gen_loss}; disc_real: {real_disc_loss}; disc_fake: {fake_disc_loss}'
				pbar.set_description(msg)
			
		if((i + 1) % (print_img_every) == 0):
			#print(f'yes')
			with torch.no_grad():
				gen.eval()
				img_labels = torch.LongTensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,1,2,3,4,5,6]).cuda()
				gen_inputs = torch.randn(32, 64).cuda()
				gen_images = gen(input = gen_inputs, labels = (img_labels - 1)).data.cpu()
				
				utils.save_image(
					gen_images,
					f'./sample_bceloss_epoch_{i + 1}.png',
					nrow = 8,
					normalize = True,
					range = (-1, 1)
				)
		if((i + 1) % save_model_every == 0):
			torch.save(
				{
				'gen': gen.module.state_dict(),
				'disc': disc.module.state_dict(),
				'gen_optimizer': gen_optimizer.state_dict(),
				'disc_optimizer': disc_optimizer.state_dict()
				},
				f'./emnist_bceloss_model_{(i + 1)}.model'
			)
	
	
if __name__ == '__main__':
	load_model = 0
	total_size = 60000
	batch_size = 128
	epoch = 200
	gen = nn.DataParallel(generator()).cuda()
	disc = nn.DataParallel(discriminator()).cuda()

	loss = nn.BCELoss()

	gen_optimizer = optim.Adam(gen.parameters(), lr = 0.0002, betas = (0.5, 0.999))
	disc_optimizer = optim.Adam(disc.parameters(), lr = 0.0002, betas = (0.5, 0.999))

	my_data = CustomDataset()
	#print(f'batch_size = {batch_size}')
	dataloader = DataLoader(my_data, batch_size = batch_size, shuffle = True)
	#print(f'len = {len(dataloader)}')
	if(load_model == 1):
		print(f'loaded model')
		checkpoint = torch.load('./emnist_bceloss_model_200.model')
		gen.module.load_state_dict(checkpoint['gen'])
		disc.module.load_state_dict(checkpoint['disc'])
		gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
		disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])

	training(epoch, gen, disc, gen_optimizer, disc_optimizer, loss, dataloader)

