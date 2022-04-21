"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from .cut_model import CUTModel
from . import networks
from torchvision.transforms import Grayscale
from kornia import color

class ProjectModel(CUTModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        CUTModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--lambda_TV', type=float, default=1.0, help='weight for TV loss')
        parser.add_argument('--lambda_identity', type=float, default=3.0, help='the "identity preservation loss"')
        parser.add_argument('--lambda_color', type=float, default=0.01, help='the "color variation weight"')
        parser.set_defaults(lr_policy="cosine",
                            lambda_GAN=1.5,
                            lambda_NCE = 3.0,
                            n_epochs=30,
                            n_epochs_decay=0,
                            lr=0.0002,
                            display_freq=100,
                            save_epoch_freq=2,
                            display_ncols=5)

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        CUTModel.__init__(self, opt)  # call the initialization method of CUTModel
        if self.opt.lambda_TV > 0.0 and self.isTrain:
            self.loss_names += ['TV']
        if self.opt.lambda_identity > 0.0 and self.isTrain:
            self.loss_names +=['idt']
            self.visual_names +=['idtl1_B']
        if self.opt.lambda_color > 0.0 and self.isTrain:
            self.loss_names +=['color']
            
    def forward(self):
        CUTModel.forward(self)
        if self.opt.lambda_identity > 0.0 and self.isTrain:
            gray_B = Grayscale(num_output_channels=3)(self.real_B)
            self.idtl1_B = self.netG(gray_B)
       
    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
        # Add TV loss
        if self.opt.lambda_TV > 0.0:
            self.loss_TV = self.compute_TV_loss() * self.opt.lambda_TV
        else:
            self.loss_TV = 0.0
        # Add identity loss
        if self.opt.lambda_identity > 0.0:
            self.loss_idt = torch.nn.functional.l1_loss(self.idtl1_B, self.real_B) * self.opt.lambda_identity
        else:
            self.loss_idt = 0.0
        # Add color loss
        if self.opt.lambda_color > 0.0:
            self.loss_color = self.compute_color_var() * self.opt.lambda_color
        else:
            self.loss_color = 0.0
        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_idt + self.loss_TV - self.loss_color
        return self.loss_G
        
    def compute_TV_loss(self):
        img = self.fake_B
        b, c, h, w = img.size()
        h_diff = torch.pow((img[:,:,1:,:]-img[:,:,:h-1,:]),2)
        h_tv = h_diff.sum()
        h_tv = h_tv / (c*(h-1)*w)
        w_tv = torch.pow((img[:,:,:,1:]-img[:,:,:,:w-1]),2).sum()
        w_tv = w_tv / (c*h*(w-1))
        return (h_tv + w_tv) * 2 / b
        
    def compute_color_var(self):
        img = self.fake_B
        _, ab = rgb2lab(img)
        b = img.size()[0]
        running_sum = 0.0
        for batch in range(b):
            ab_mat = torch.flatten(ab[batch], 1, 2)
            cov = torch.cov(ab_mat)
            running_sum += torch.sqrt(torch.min(cov[0,0], cov[1,1]))
        running_sum /= b
        return running_sum
        
        

def rgb2lab(rgb):
    #Denormalize
    rgb = (rgb + 1) * 0.5
    lab = color.rgb_to_lab(rgb)
    l, ab = lab[:,[0]], lab[:,[1,2]]
    return l, ab
    
