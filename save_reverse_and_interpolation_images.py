from __future__ import print_function
import argparse
import logging
import sys, os
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from dataset import get_train_loader, get_test_loader
from eval_metrics import eval_sysu, eval_regdb
from utils import *
import pdb
import scipy.io as scio
from models import Flow2Flow
import warnings
warnings.filterwarnings("ignore")


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu')
parser.add_argument('--data_path', default='../data/sysu', help='dataset path')
parser.add_argument('--sample_method', default='identity_random', type=str, help='method to sample images')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test_only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--model_prefix', type=str, default='Flow2Flow-one_activation', help='prefix for saved model')
parser.add_argument('--constrain_feat', action='store_true', help='whether to use z loss')
parser.add_argument('--constrain_identity', action='store_true', help='whether to use identity loss')
parser.add_argument('--constrain_modality', action='store_true', help='whether to use modal loss')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--img_w', default=72, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=144, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch_size', default=3, type=int, metavar='B', help='training batch size')
parser.add_argument('--test_batch', default=4, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str, metavar='m', help='method type: base or agw, adp')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=8, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--total_epoch', default=200, type=int, metavar='t', help='random seed')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--augc', default=0 , type=int, metavar='aug', help='use channel aug or not')
parser.add_argument('--rande', default= 0 , type=float, metavar='ra', help='use random erasing or not')
parser.add_argument('--kl', default= 0 , type=float, metavar='kl', help='use kl loss and the weight')
parser.add_argument('--alpha', default=1 , type=int, metavar='alpha', help='magnification for the hard mining')
parser.add_argument('--gamma', default=1 , type=int, metavar='gamma', help='gamma for the hard mining')
parser.add_argument('--square', default= 1 , type=int, metavar='square', help='gamma for the hard mining')
parser.add_argument('--is_training', default=True, type=bool, help='training or testing')
parser.add_argument('--initializer', default='normal', type=str, help='method to initialize flow models')
parser.add_argument('--weight_norm_l2', type=float, default=5e-5, help='L2 regularization factor for weight norm')
parser.add_argument('--lr_G', default=0.0001 , type=float, help='learning rate')
parser.add_argument('--lr_S', default=0.0001 , type=float, help='learning rate')
parser.add_argument('--beta_1', default=0.5 , type=float, help='beta_1 for adam')
parser.add_argument('--beta_2', default=0.999 , type=float, help='beta_2 for adam')
parser.add_argument('--lr_policy', type=str, default='linear', help='Learning rate schedule policy', choices=('linear', 'plateau', 'step'))
parser.add_argument('--lr_step_epochs', type=int, default=100, help='Number of epochs between each divide-by-10 step (step policy only).')
parser.add_argument('--lr_warmup_epochs', type=int, default=100, help='Number of epochs before we start decaying the learning rate (linear only).')
parser.add_argument('--lr_decay_epochs', type=int, default=100, help='Number of epochs to decay the learning rate linearly to 0 (linear only).')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--log_file', type=str, default='log.txt')
#### flow model parameters
parser.add_argument('--in_channel', default=3, type=int, help='input channel')
parser.add_argument('--sim_feat', default=512, type=int, help='output channel of similarity net')
parser.add_argument('--n_flow', default=10, type=int, help='flow numbers')
parser.add_argument('--n_block', default=1, type=int, help='block numbers')
parser.add_argument('--squeeze_fold', default=4, type=int, help='squeeze fold')
#### flow2flow new parameters
parser.add_argument('--norm_type', type=str, default='instance', choices=('instance', 'batch', 'group'))
parser.add_argument('--num_channels', default=3, type=int, help='Number of channels in an image.')
parser.add_argument('--num_channels_d', default=64, type=int, help='Number of filters in the discriminator.')
parser.add_argument('--num_channels_g', default=32, type=int, help='Number of filters in the generator.')
parser.add_argument('--num_blocks', default=4, type=int, help='Number of blocks in the STResNet.')
parser.add_argument('--num_scales', default=2, type=int, help='Scale factor.')
parser.add_argument('--kernel_size_d', default=4, type=int, help='Size of the discriminator\'s kernels.')

parser.add_argument('--epoch', default=42, type=int, help='loading epoch')


#### 1 training args and logger settings
args = parser.parse_args()
set_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if os.path.exists(args.log_path):
    pass
else:
    os.mkdir(args.log_path)
logger = get_logger(os.path.join(args.log_path, args.log_file))
logger.info("==========\nArgs:{}\n==========".format(args))

#### 2 initialize dataset
train_loader = get_train_loader(dataset=args.dataset,
                                root=args.data_path,
                                sample_method=args.sample_method,
                                batch_size=args.batch_size*args.num_pos,
                                p_size=args.batch_size,
                                k_size=args.num_pos,
                                random_flip=False,
                                random_crop=False,
                                random_erase=False,
                                color_jitter=False,
                                padding=0,
                                image_size=(args.img_h, args.img_w),
                                num_workers=args.workers)
# visible images  infrared images
gallery_loader, query_loader = get_test_loader(dataset=args.dataset,
                                               root=args.data_path,
                                               batch_size=args.batch_size,
                                               image_size=(args.img_h, args.img_w),
                                               num_workers=args.workers)
logger.info("--->>> length of train loader   %d " % (len(train_loader)))
logger.info("--->>> length of gallery loader %d " % (len(gallery_loader)))
logger.info("--->>> length of query loader   %d " % (len(query_loader)))

#### 3 initialize flow model
FlowModel = Flow2Flow(args)
epoch = args.epoch
path_models = args.model_prefix + '-epoch_' + str(epoch) + '_models.pth'
checkpoint = torch.load(os.path.join(args.model_path, path_models))
FlowModel.load_state_dict(checkpoint['model'])
FlowModel.eval()
logger.info("--->>> load models of epoch %d successfully!" % (epoch))


# ### test interpolation
# save_dir = './save_interpolation_images'
# if os.path.exists(save_dir):
#     pass
# else:
#     os.makedirs(save_dir)
# for idx, (img, label, cam, path, item) in enumerate(train_loader):
#     print('train loader idx ', idx)
#     # preprocessing
#     bs = img.shape[0]
#     ir_idx = list(range(0, bs, 2))
#     rgb_idx = list(range(1, bs, 2))
#     img_rgb = img[rgb_idx]
#     img_ir = img[ir_idx]
#     target_rgb = label[rgb_idx]
#     target_ir = label[ir_idx]
#     path_rgb = [path[i] for i in rgb_idx]
#     path_ir = [path[i] for i in ir_idx]
#     # network propagation
#     img_rgb, img_ir = img_rgb.cuda(), img_ir.cuda()
#     target_rgb, target_ir = target_rgb.cuda(), target_ir.cuda()
#     FlowModel.set_inputs(img_rgb, img_ir, target_rgb, target_ir)
#     # generate ir
#     rgb2lat, _ = FlowModel.g_rgb(FlowModel.rgb, reverse=False)
#     rgb2lat_0 = rgb2lat[0]
#     rgb2lat_1 = rgb2lat[3]
#     rgb2lat_fake1 = rgb2lat_0 + (rgb2lat_1-rgb2lat_0) / 10 * 1
#     rgb2lat_fake2 = rgb2lat_0 + (rgb2lat_1-rgb2lat_0) / 10 * 2
#     rgb2lat_fake3 = rgb2lat_0 + (rgb2lat_1-rgb2lat_0) / 10 * 3
#     rgb2lat_fake4 = rgb2lat_0 + (rgb2lat_1-rgb2lat_0) / 10 * 4
#     rgb2lat_fake5 = rgb2lat_0 + (rgb2lat_1-rgb2lat_0) / 10 * 5
#     rgb2lat_fake6 = rgb2lat_0 + (rgb2lat_1-rgb2lat_0) / 10 * 6
#     rgb2lat_fake7 = rgb2lat_0 + (rgb2lat_1-rgb2lat_0) / 10 * 7
#     rgb2lat_fake8 = rgb2lat_0 + (rgb2lat_1-rgb2lat_0) / 10 * 8
#     rgb2lat_fake9 = rgb2lat_0 + (rgb2lat_1-rgb2lat_0) / 10 * 9
#     rgb2lat_v = torch.cat((rgb2lat_0.unsqueeze(0),
#                            rgb2lat_fake1.unsqueeze(0),
#                            rgb2lat_fake2.unsqueeze(0),
#                            rgb2lat_fake3.unsqueeze(0),
#                            rgb2lat_fake4.unsqueeze(0),
#                            rgb2lat_fake5.unsqueeze(0),
#                            rgb2lat_fake6.unsqueeze(0),
#                            rgb2lat_fake7.unsqueeze(0),
#                            rgb2lat_fake8.unsqueeze(0),
#                            rgb2lat_fake9.unsqueeze(0),
#                            rgb2lat_1.unsqueeze(0)), dim=0)
#     rgb_interpolation, _ = FlowModel.g_rgb(rgb2lat_v, reverse=True)
#     rgb_interpolation = torch.tanh(rgb_interpolation)
#     rgb_interpolation = un_normalize(rgb_interpolation).detach().cpu()
#     for i in range(rgb_interpolation.shape[0]):
#         interpolation_path = os.path.join(save_dir, str(i)+'.jpg')
#         interpolation_i = torch.cat((rgb_interpolation[i][2,:,:].unsqueeze(0), \
#                                      rgb_interpolation[i][1,:,:].unsqueeze(0), \
#                                      rgb_interpolation[i][0,:,:].unsqueeze(0)), dim=0)
#         interpolation_i = interpolation_i.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
#         cv2.imwrite(interpolation_path, interpolation_i)
#     if idx>4:
#         break


### test noise
save_dir = './save_noise_images'
if os.path.exists(save_dir):
    pass
else:
    os.makedirs(save_dir)

for idx, (img, label, cam, path, item) in enumerate(train_loader):
    print('train loader idx ', idx)
    # preprocessing
    bs = img.shape[0]
    ir_idx = list(range(0, bs, 2))
    rgb_idx = list(range(1, bs, 2))
    img_rgb = img[rgb_idx]
    img_ir = img[ir_idx]
    target_rgb = label[rgb_idx]
    target_ir = label[ir_idx]
    path_rgb = [path[i] for i in rgb_idx]
    path_ir = [path[i] for i in ir_idx]
    # network propagation
    img_rgb, img_ir = img_rgb.cuda(), img_ir.cuda()
    target_rgb, target_ir = target_rgb.cuda(), target_ir.cuda()
    FlowModel.set_inputs(img_rgb, img_ir, target_rgb, target_ir)
    # generate ir
    rgb2lat, _ = FlowModel.g_rgb(FlowModel.rgb, reverse=False)
    rgb2lat_0 = rgb2lat[0]
    z_dim = 3*args.img_h*args.img_w
    noise = np.random.normal(0, 1, (z_dim))
    noise = torch.from_numpy(noise).float().to('cuda').reshape(3, args.img_h, args.img_w).cuda()
    rgb2lat_fake1 = (rgb2lat_0*99 + noise*1) / 100
    rgb2lat_fake2 = (rgb2lat_0*98 + noise*2) / 100
    rgb2lat_fake3 = (rgb2lat_0*7 + noise*3) / 10
    rgb2lat_fake4 = (rgb2lat_0*6 + noise*4) / 10
    rgb2lat_fake5 = (rgb2lat_0*5 + noise*5) / 10
    rgb2lat_fake6 = (rgb2lat_0*4 + noise*6) / 10
    rgb2lat_fake7 = (rgb2lat_0*3 + noise*7) / 10
    rgb2lat_fake8 = (rgb2lat_0*2 + noise*8) / 10
    rgb2lat_fake9 = (rgb2lat_0*1 + noise*9) / 10


    rgb2lat_v = torch.cat((rgb2lat_0.unsqueeze(0),
                           rgb2lat_fake1.unsqueeze(0),
                           rgb2lat_fake2.unsqueeze(0),
                           rgb2lat_fake3.unsqueeze(0),
                           rgb2lat_fake4.unsqueeze(0),
                           rgb2lat_fake5.unsqueeze(0),
                           rgb2lat_fake6.unsqueeze(0),
                           rgb2lat_fake7.unsqueeze(0),
                           rgb2lat_fake8.unsqueeze(0),
                           rgb2lat_fake9.unsqueeze(0)), dim=0)
    rgb_interpolation, _ = FlowModel.g_rgb(rgb2lat_v, reverse=True)
    rgb_interpolation = torch.tanh(rgb_interpolation)
    rgb_interpolation = un_normalize(rgb_interpolation).detach().cpu()
    for i in range(rgb_interpolation.shape[0]):
        interpolation_path = os.path.join(save_dir, str(i)+'.jpg')
        interpolation_i = torch.cat((rgb_interpolation[i][2,:,:].unsqueeze(0), \
                                     rgb_interpolation[i][1,:,:].unsqueeze(0), \
                                     rgb_interpolation[i][0,:,:].unsqueeze(0)), dim=0)
        interpolation_i = interpolation_i.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        cv2.imwrite(interpolation_path, interpolation_i)
    if idx>4:
        break



# ### test inverse
# save_dir = './save_reverse_images'
# for idx, (img, label, cam, path, item) in enumerate(train_loader):
#     print('train loader idx ', idx)
#     # preprocessing
#     bs = img.shape[0]
#     ir_idx = list(range(0, bs, 2))
#     rgb_idx = list(range(1, bs, 2))
#     img_rgb = img[rgb_idx]
#     img_ir = img[ir_idx]
#     target_rgb = label[rgb_idx]
#     target_ir = label[ir_idx]
#     path_rgb = [path[i] for i in rgb_idx]
#     path_ir = [path[i] for i in ir_idx]
#     # network propagation
#     img_rgb, img_ir = img_rgb.cuda(), img_ir.cuda()
#     target_rgb, target_ir = target_rgb.cuda(), target_ir.cuda()
#     FlowModel.set_inputs(img_rgb, img_ir, target_rgb, target_ir)
#     # generate ir
#     rgb2lat, _ = FlowModel.g_rgb(FlowModel.rgb, reverse=False)
#     rgb2rgb, _ = FlowModel.g_rgb(rgb2lat, reverse=True)
#     rgb2rgb = torch.tanh(rgb2rgb)
#     # generate rgb
#     ir2lat, _ = FlowModel.g_ir(FlowModel.ir, reverse=False)
#     ir2ir, _ = FlowModel.g_ir(ir2lat, reverse=True)
#     ir2ir = torch.tanh(ir2ir)

#     img_rgb, img_ir = un_normalize(img_rgb).detach().cpu(), un_normalize(img_ir).detach().cpu()
#     rgb2lat, ir2lat = rgb2lat.detach().cpu(), ir2lat.detach().cpu()
#     rgb2rgb, ir2ir = un_normalize(rgb2rgb).detach().cpu(), un_normalize(ir2ir).detach().cpu()

#     # save images
#     num_images_one_iter = len(path_rgb)
#     for i in range(num_images_one_iter):
#         # for the true visible images
#         rgb_name = path_rgb[i].split('/')        # '../data/sysu/cam5/0340/0008.jpg'
#         rgb_dir = os.path.join(save_dir, 'true', rgb_name[-3], rgb_name[-2])
#         if os.path.exists(rgb_dir):
#             pass
#         else:
#             os.makedirs(rgb_dir)
#         rgb_path = os.path.join(rgb_dir, rgb_name[-1])
#         # for the true infrared images
#         ir_name = path_ir[i].split('/')        # '../data/sysu/cam5/0340/0008.jpg'
#         ir_dir = os.path.join(save_dir, 'true', ir_name[-3], ir_name[-2])
#         if os.path.exists(ir_dir):
#             pass
#         else:
#             os.makedirs(ir_dir)
#         ir_path = os.path.join(ir_dir, ir_name[-1])
#         # for the generated visible images
#         ir2ir_dir = os.path.join(save_dir, 'fake', ir_name[-3], ir_name[-2])
#         if os.path.exists(ir2ir_dir):
#             pass
#         else:
#             os.makedirs(ir2ir_dir)
#         ir2ir_path = os.path.join(ir2ir_dir, ir_name[-1])
#         # for the generated infrared images
#         rgb2rgb_dir = os.path.join(save_dir, 'fake', rgb_name[-3], rgb_name[-2])
#         if os.path.exists(rgb2rgb_dir):
#             pass
#         else:
#             os.makedirs(rgb2rgb_dir)
#         rgb2rgb_path = os.path.join(rgb2rgb_dir, rgb_name[-1])

#         rgb_i = torch.cat((img_rgb[i][2,:,:].unsqueeze(0), \
#                            img_rgb[i][1,:,:].unsqueeze(0), \
#                            img_rgb[i][0,:,:].unsqueeze(0)), dim=0)
#         rgb_i = rgb_i.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
#         ir_i = torch.cat((img_ir[i][2,:,:].unsqueeze(0), \
#                            img_ir[i][1,:,:].unsqueeze(0), \
#                            img_ir[i][0,:,:].unsqueeze(0)), dim=0)
#         ir_i = ir_i.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
#         ir2ir_i = torch.cat((ir2ir[i][2,:,:].unsqueeze(0), \
#                              ir2ir[i][1,:,:].unsqueeze(0), \
#                              ir2ir[i][0,:,:].unsqueeze(0)), dim=0)
#         ir2ir_i = ir2ir_i.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
#         rbg2rgb_i = torch.cat((rgb2rgb[i][2,:,:].unsqueeze(0), \
#                                rgb2rgb[i][1,:,:].unsqueeze(0), \
#                                rgb2rgb[i][0,:,:].unsqueeze(0)), dim=0)
#         rbg2rgb_i = rbg2rgb_i.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
#         cv2.imwrite(rgb_path, rgb_i)
#         cv2.imwrite(ir_path, ir_i)
#         cv2.imwrite(ir2ir_path, ir2ir_i)
#         cv2.imwrite(rgb2rgb_path, rbg2rgb_i)
#     if idx>5:
#         break

