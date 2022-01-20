import argparse
import sys
import os
import cv2
import numpy as np
import torch

from tqdm.auto import trange
from torch import nn, optim
from torch.nn import functional as F
from torch.backends import cudnn


from models import Autoencoder, toTensor, var_to_np
from util import get_image_paths, load_images, stack_images
from training_data import get_training_data

def main(args):
    parser = argparse.ArgumentParser(description='DeepFake-Pytorch')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100000,
                        help='number of epochs to train (default: 10000)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--crawling', action='store_true')
    try:
        opt = parser.parse_args(args[1:])
    except:
        parser.print_help()
        raise

    # Environment settings
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        print('===> Using GPU to train')
        device = torch.device('cuda:0')
        cudnn.benchmark = True
        torch.cuda.manual_seed(opt.seed)
    else:
        print('===> Using CPU to train')
        device = "cpu"


    # Load png/jpg files
    print('===> Loading datasets')
    if opt.crawling:
        from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
        crawler = BingImageCrawler(storage={'root_dir':'data/holland'}, feeder_threads=2, parser_threads=2, downloader_threads=4)
        crawler.crawl(keyword='Tom Holland', max_num=1000)
        crawler = BingImageCrawler(storage={'root_dir':'data/chalamet'}, feeder_threads=2, parser_threads=2, downloader_threads=4)
        crawler.crawl(keyword='Timothee Chalamet ', max_num=1000)
        return
    images_A = get_image_paths("data/holland")
    images_B = get_image_paths("data/chalamet")
    
    # Resize input files 
    images_A = load_images(images_A) / 255.0
    images_B = load_images(images_B) / 255.0
    # scale mean value of images_A per channel 
    images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))
    
    # Model settings
    start_epoch = 0
    criterion = nn.L1Loss()
    model = Autoencoder().to(device)
    optimizer_1 = optim.Adam([{'params': model.encoder.parameters()},
                              {'params': model.decoder_A.parameters()}]
                             , lr=5e-5, betas=(0.5, 0.999))
    optimizer_2 = optim.Adam([{'params': model.encoder.parameters()},
                              {'params': model.decoder_B.parameters()}]
                             , lr=5e-5, betas=(0.5, 0.999))

    if os.path.isdir('checkpoint'):
        try:
            checkpoint = torch.load('./checkpoint/autoencoder.t7')
        except FileNotFoundError:
            print('Can\'t found autoencoder.t7')
            raise
        model.load_state_dict(checkpoint['state'])
        start_epoch = checkpoint['epoch']
        optimizer_1.load_state_dict(checkpoint['optimizer1_state_dict'])
        optimizer_2.load_state_dict(checkpoint['optimizer2_state_dict'])
    
    print('===> Start from last epoch: {}'.format(start_epoch))
    for epoch in trange(start_epoch, opt.epochs):
        batch_size = opt.batch_size

        warped_A, target_A = get_training_data(images_A, batch_size)
        warped_B, target_B = get_training_data(images_B, batch_size)

        warped_A, target_A = toTensor(warped_A), toTensor(target_A)
        warped_B, target_B = toTensor(warped_B), toTensor(target_B)

        warped_A = warped_A.to(device).float()
        target_A = target_A.to(device).float()
        warped_B = warped_B.to(device).float()
        target_B = target_B.to(device).float()

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        warped_A = model(warped_A, 'A')
        warped_B = model(warped_B, 'B')

        loss1 = criterion(warped_A, target_A)
        loss2 = criterion(warped_B, target_B)
        loss1.backward()
        loss2.backward()
        optimizer_1.step()
        optimizer_2.step()

        if epoch % opt.log_interval == 0:
            test_A_ = target_A[0:14]
            test_B_ = target_B[0:14]
            test_A = var_to_np(target_A[0:14])
            test_B = var_to_np(target_B[0:14])
            state = {
                'state': model.state_dict(),
                'epoch': epoch,
                'optimizer1_state_dict': optimizer_1.state_dict(),
                'optimizer2_state_dict': optimizer_2.state_dict(),
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/autoencoder.t7')

        # original files, autoencoded files, faceswapped files
        figure_A = np.stack([
            test_A,
            var_to_np(model(test_A_, 'A')),
            var_to_np(model(test_A_, 'B')),
        ], axis=1)
        figure_B = np.stack([
            test_B,
            var_to_np(model(test_B_, 'B')),
            var_to_np(model(test_B_, 'A')),
        ], axis=1)

        figure = np.concatenate([figure_A, figure_B], axis=0)
        figure = figure.transpose((0, 1, 3, 4, 2))
        figure = figure.reshape((4, 7) + figure.shape[1:])
        figure = stack_images(figure)

        figure = np.clip(figure * 255, 0, 255).astype('uint8')
        cv2.imwrite('output.jpg', figure)

if __name__ == "__main__":
    main(sys.argv)
