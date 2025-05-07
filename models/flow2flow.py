import os
import torch
import torch.nn as nn
import util
from itertools import chain
from .patch_gan import PatchGAN
from torch.autograd import Variable
from models.real_nvp import RealNVP, RealNVPLoss


class Flow2Flow(nn.Module):
    def __init__(self, args):
        super(Flow2Flow, self).__init__()
        self.device = 'cuda' if len(args.gpu) > 0 else 'cpu'
        self.gpu = [0] if len(args.gpu)==1 else [int(i) for i in args.gpu.split(',')]
        self.is_training = args.is_training
        self.in_channels = args.num_channels
        self.num_channels_g = args.num_channels_g
        self.num_blocks = args.num_blocks
        self.num_scales = args.num_scales
        self.out_channels = 4 ** (self.num_scales - 1) * self.in_channels
        # Set up RealNVP generators
        self.g_rgb = RealNVP(num_scales=self.num_scales,         # 2
                             in_channels=self.in_channels,       # 3
                             mid_channels=self.num_channels_g,   # 32
                             num_blocks=self.num_blocks,         # 4
                             un_normalize_x=True,
                             no_latent=False)
        util.init_model(self.g_rgb, init_method=args.initializer)
        self.g_ir = RealNVP(num_scales=self.num_scales,
                             in_channels=self.in_channels,
                             mid_channels=self.num_channels_g,
                             num_blocks=self.num_blocks,
                             un_normalize_x=True,
                             no_latent=False)
        util.init_model(self.g_ir, init_method=args.initializer)
        if self.is_training:
            # Set up discriminators
            self.e_rgb = PatchGAN(args, return_binary=False)
            self.e_ir = PatchGAN(args, return_binary=False)
            self.d_rgb = PatchGAN(args, return_binary=True)  # Answers Q "is this rgb image real?"
            self.d_ir = PatchGAN(args, return_binary=True)   # Answers Q "is this ir image real?"
            self._data_parallel()
            # Set up loss functions
            self.max_grad_norm = 0.0
            self.lambda_mle = 0.0001
            self.mle_loss_fn = RealNVPLoss()           # to maximize log_det
            self.mle_loss_fn = self.mle_loss_fn.to(self.device)
            # self.gan_loss_fn = util.GANLoss(device=self.device, use_least_squares=True)
            self.id_loss_fn = util.CrossModalityIdentityLoss()
            self.id_loss_fn = self.id_loss_fn.to(self.device)
            self.tri_loss_fn = util.TripletLoss()
            self.tri_loss_fn = self.tri_loss_fn.to(self.device)
            self.latent_loss_fn = util.HiddenFeatureLoss()
            self.latent_loss_fn = self.latent_loss_fn.to(self.device)
            self.modality_loss_fn = torch.nn.MSELoss()
            self.modality_loss_fn = self.modality_loss_fn.to(self.device)
            Tensor = torch.cuda.FloatTensor if self.device=='cuda' else torch.Tensor
            self.target_real = Variable(Tensor(args.batch_size).fill_(1.0), requires_grad=False)
            self.target_fake = Variable(Tensor(args.batch_size).fill_(0.0), requires_grad=False)
            # Set up optimizers
            g_rgb_params = util.get_param_groups(self.g_rgb, 5e-5, norm_suffix='weight_g')
            g_ir_params = util.get_param_groups(self.g_ir, 5e-5, norm_suffix='weight_g')
            self.opt_g = torch.optim.Adam(chain(g_rgb_params, g_ir_params), lr=2e-4, betas=(0.5, 0.999))
            self.opt_d = torch.optim.Adam(chain(self.e_ir.parameters(), self.e_rgb.parameters(),
                                                self.d_ir.parameters(), self.d_rgb.parameters()),
                                          lr=2e-4, betas=(0.5, 0.999))
            self.optimizers = [self.opt_g, self.opt_d]
            self.schedulers = [util.get_lr_scheduler(opt, args) for opt in self.optimizers]
            # Setup image mixers
            buffer_capacity = 50
            self.rgb2ir_buffer = util.ImageBuffer(buffer_capacity)  # Buffer of generated ir images
            self.ir2rgb_buffer = util.ImageBuffer(buffer_capacity)  # Buffer of generated rgb images
        else:
            self._data_parallel()

        # Images in flow rgb -> lat -> ir
        self.rgb = None
        self.rgb2lat = None
        self.rgb2ir = None
        self.rgb_target = None
        # Images in flow ir -> lat -> rgb
        self.ir = None
        self.ir2lat = None
        self.ir2rgb = None
        self.ir_target = None
        # Discriminator losses
        self.loss_e_id_rgb = None
        self.loss_e_id_ir = None
        self.loss_e_id = None
        self.loss_d_mod_rgb = None
        self.loss_d_mod_ir = None
        self.loss_d_mod = None
        self.loss_tri_rgb = None
        self.loss_tri_ir = None
        self.loss_tri = None
        self.loss_d = None
        # Generator losses
        self.loss_mle_rgb = None
        self.loss_mle_ir = None
        self.loss_mle = None
        self.loss_g_id_rgb = None
        self.loss_g_id_ir = None
        self.loss_g_id = None
        self.loss_g_mod_rgb = None
        self.loss_g_mod_ir = None
        self.loss_g_mod = None
        self.loss_feat = None
        self.loss_g = None

    def set_inputs(self, rgb_input, ir_input, rgb_target, ir_target):
        """Set the inputs prior to a forward pass through the network.
        Args:
            rgb_input: Tensor with rgb input
            ir_input: Tensor with ir input
        """
        self.rgb = rgb_input.to(self.device)
        self.ir = ir_input.to(self.device)
        self.rgb_target = rgb_target.to(self.device)
        self.ir_target = ir_target.to(self.device)
        # print('self.rgb -> ', self.rgb.shape)
        # print('self.ir -> ', self.ir.shape)
        # print('self.rgb_target -> ', self.rgb_target.shape)
        # print('self.ir_target -> ', self.ir_target.shape)

    def _data_parallel(self):
        self.g_rgb = nn.DataParallel(self.g_rgb, device_ids=self.gpu).to(self.device)
        self.g_ir = nn.DataParallel(self.g_ir, device_ids=self.gpu).to(self.device)
        if self.is_training:
            self.e_rgb = nn.DataParallel(self.e_rgb, self.gpu).to(self.device)
            self.e_ir = nn.DataParallel(self.e_ir, self.gpu).to(self.device)
            self.d_rgb = nn.DataParallel(self.d_rgb, self.gpu).to(self.device)
            self.d_ir = nn.DataParallel(self.d_ir, self.gpu).to(self.device)

    def backward_d(self):
        # identity loss
        bs = self.rgb.shape[0]
        feat_rgb = self.e_rgb(torch.cat((self.rgb, self.ir2rgb.detach()), dim=0))
        feat_rgb_real = feat_rgb[:bs]
        feat_rgb_fake = feat_rgb[bs:]
        self.loss_e_id_rgb = 2.0 - self.id_loss_fn(feat_rgb_real, feat_rgb_fake, self.rgb_target, self.ir_target)
        feat_ir = self.e_ir(torch.cat((self.ir, self.rgb2ir.detach()), dim=0))
        feat_ir_real = feat_ir[:bs]
        feat_ir_fake = feat_ir[bs:]
        self.loss_e_id_ir = 2.0 - self.id_loss_fn(feat_ir_real, feat_ir_fake, self.ir_target, self.rgb_target)
        self.loss_e_id = self.loss_e_id_rgb + self.loss_e_id_ir
        # # triplet loss
        # self.loss_tri_rgb = self.tri_loss_fn(feat_rgb_real, self.rgb_target)
        # self.loss_tri_ir = self.tri_loss_fn(feat_ir_real, self.ir_target)
        # self.loss_tri = self.loss_tri_rgb + self.loss_tri_ir
        # modality loss
        self.loss_d_mod_rgb = self.modality_loss_fn(self.d_rgb(self.rgb), self.target_real) + \
                              self.modality_loss_fn(self.d_rgb(self.ir2rgb.detach()), self.target_fake)
        self.loss_d_mod_ir = self.modality_loss_fn(self.d_ir(self.ir), self.target_real) + \
                             self.modality_loss_fn(self.d_ir(self.rgb2ir.detach()), self.target_fake)
        self.loss_d_mod = self.loss_d_mod_rgb + self.loss_d_mod_ir
        # total loss
        self.loss_d = self.loss_e_id  + self.loss_d_mod
        # self.loss_d = self.loss_e_id + self.loss_tri + self.loss_d_mod
        self.loss_d.backward()
        # print('self.loss_e_id -> ', self.loss_e_id)
        # print('self.loss_tri -> ', self.loss_tri)
        # print('self.loss_d_mod -> ', self.loss_d_mod)
        # print()

    def backward_g(self):
        # MLE loss
        self.rgb2lat, sldj_rgb2lat = self.g_rgb(self.rgb, reverse=False)
        rgb2ir, _ = self.g_ir(self.rgb2lat, reverse=True)
        self.rgb2ir = torch.tanh(rgb2ir)
        self.loss_mle_rgb = self.lambda_mle * self.mle_loss_fn(self.rgb2lat, sldj_rgb2lat)
        self.ir2lat, sldj_ir2lat = self.g_ir(self.ir, reverse=False)
        # Finish ir -> lat -> rgb: Say source is real to invert loss
        ir2rgb, _ = self.g_rgb(self.ir2lat, reverse=True)
        self.ir2rgb = torch.tanh(ir2rgb)
        self.loss_mle_ir = self.lambda_mle * self.mle_loss_fn(self.ir2lat, sldj_ir2lat)
        self.loss_mle = self.loss_mle_rgb + self.loss_mle_ir
        # hidden feature loss
        bs = self.rgb.shape[0]
        feats = torch.cat((self.rgb2lat.reshape(bs, -1), self.ir2lat.reshape(bs, -1)), dim=0)
        targets = torch.cat((self.rgb_target, self.ir_target), dim=0)
        self.loss_feat = self.latent_loss_fn(feats, targets)
        # identity loss
        feat_rgb = self.e_rgb(torch.cat((self.rgb, self.ir2rgb), dim=0))
        feat_rgb_real = feat_rgb[:bs]
        feat_rgb_fake = feat_rgb[bs:]
        self.loss_g_id_rgb = self.id_loss_fn(feat_rgb_real, feat_rgb_fake, self.rgb_target, self.ir_target)
        feat_ir = self.e_ir(torch.cat((self.ir, self.rgb2ir), dim=0))
        feat_ir_real = feat_ir[:bs]
        feat_ir_fake = feat_ir[bs:]
        self.loss_g_id_ir = self.id_loss_fn(feat_ir_real, feat_ir_fake, self.ir_target, self.rgb_target)
        self.loss_g_id = self.loss_g_id_rgb + self.loss_g_id_ir
        # modality loss
        self.loss_g_mod_rgb = self.modality_loss_fn(self.d_rgb(self.ir2rgb), self.target_real)
        self.loss_g_mod_ir = self.modality_loss_fn(self.d_ir(self.rgb2ir), self.target_real)
        self.loss_g_mod = self.loss_g_mod_rgb + self.loss_g_mod_ir
        # Total losses
        self.loss_g = self.loss_mle + 0.01*self.loss_feat + self.loss_g_id + self.loss_g_mod
        self.loss_g.backward()
        # print('self.loss_mle -> ', self.loss_mle)
        # print('self.loss_feat -> ', self.loss_feat)
        # print('self.loss_g_id -> ', self.loss_g_id)
        # print('self.loss_g_mod -> ', self.loss_g_mod)

    def train_iter(self):
        # Backprop the generators
        self.opt_g.zero_grad()
        self.backward_g()
        util.clip_grad_norm(self.opt_g, self.max_grad_norm)
        self.opt_g.step()
        # Backprop the discriminators
        self.opt_d.zero_grad()
        self.backward_d()
        util.clip_grad_norm(self.opt_d, self.max_grad_norm)
        self.opt_d.step()
        return self.loss_g.detach().cpu(), self.loss_d.detach().cpu()



