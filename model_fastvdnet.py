
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision

class FastVDnet(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()
        self.config = config
        self.channels = config.no_in_channel
        self.num_input_frames = 5
        # Define models of each denoising stage
        self.temp1 = DenBlock(num_input_frames=3*self.channels,outchannel=1)
        self.temp2 = DenBlock(num_input_frames=3*self.channels,outchannel=1)
        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        '''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
        (x0, x1, x2, x3, x4) = tuple(x[:, self.channels*m:self.channels*m+self.channels, :, :] for m in range(self.num_input_frames))

        # First stage
        x20 = self.temp1(x0, x1, x2)
        x21 = self.temp1(x1, x2, x3)
        x22 = self.temp1(x2, x3, x4)

        #Second stage
        x = self.temp2(x20, x21, x22)

        # if(self.channels>1):
        #     x = torch.cat((x, x*0), dim=1) # adding second dim for FM imaging complex processing
        
        return x

class FastVDnet_7(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()
        self.config = config
        self.channels = config.no_in_channel
        self.num_input_frames = 7
        # Define models of each denoising stage
        self.temp1 = DenBlock(num_input_frames=3*self.channels,outchannel=1)
        self.temp2 = FastVDnet(config)
        
        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        '''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
        (x0, x1, x2, x3, x4, x5, x6) = tuple(x[:, self.channels*m:self.channels*m+self.channels:2, :, :] for m in range(self.num_input_frames))

        # First stage
        x20 = self.temp1(x0, x1, x2)
        x21 = self.temp1(x1, x2, x3)
        x22 = self.temp1(x2, x3, x4)
        x23 = self.temp1(x3, x4, x5)
        x24 = self.temp1(x4, x5, x6)

        #Second stage
        x   = self.temp2(torch.cat((x20, x21, x22, x23, x24),axis=1))

        # if(self.channels>1):
        #     x = torch.cat((x, x*0), dim=1) # adding second dim for FM imaging complex processing
        
        return x

class FastVDnet_9(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()
        self.config = config
        self.channels = config.no_in_channel
        self.num_input_frames = 9
        # Define models of each denoising stage
        self.temp1 = DenBlock(num_input_frames=3*self.channels,outchannel=1)
        self.temp2 = FastVDnet_7(config)
        
        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        '''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
        (x0, x1, x2, x3, x4, x5, x6, x7, x8) = tuple(x[:, self.channels*m:self.channels*m+self.channels:2, :, :] for m in range(self.num_input_frames))

        # First stage
        x20 = self.temp1(x0, x1, x2)
        x21 = self.temp1(x1, x2, x3)
        x22 = self.temp1(x2, x3, x4)
        x23 = self.temp1(x3, x4, x5)
        x24 = self.temp1(x4, x5, x6)
        x25 = self.temp1(x5, x6, x7)
        x26 = self.temp1(x6, x7, x8)
        

        #Second stage
        x   = self.temp2(torch.cat((x20, x21, x22, x23, x24, x25, x26),axis=1))


        # if(self.channels>1):
        #     x = torch.cat((x, x*0), dim=1) # adding second dim for FM imaging complex processing
        
        return x

class CvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''
    def __init__(self, in_ch, out_ch):
       super(CvBlock, self).__init__()
       self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			#nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			#nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

    def forward(self, x):
        return self.convblock(x)

class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
    def __init__(self, num_in_frames, out_ch):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 32
        self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames, num_in_frames*self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			#nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			#nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

    def forward(self, x):
        return self.convblock(x)

class DownBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(DownBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
			#nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			CvBlock(out_ch, out_ch)
		)

	def forward(self, x):
		return self.convblock(x)

class UpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(UpBlock, self).__init__()
		self.convblock = nn.Sequential(
			CvBlock(in_ch, in_ch),
			nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
			nn.PixelShuffle(2)
		)

	def forward(self, x):
		return self.convblock(x)

class OutputCvBlock(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			#nn.BatchNorm2d(in_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
		)

	def forward(self, x):
		return self.convblock(x)

class DenBlock(nn.Module):
    """ Definition of the denosing block of FastDVDnet.
    Inputs of constructor:
        num_input_frames: int. number of input frames
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, num_input_frames=3,outchannel=2):
        super(DenBlock, self).__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128

        self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=outchannel)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, in0, in1, in2):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
        '''
        # Input convolution block
        x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
        # Downsampling
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        # Upsampling
        x2 = self.upc2(x2)
        x1 = self.upc1(x1+x2)
        # Estimation
        x = self.outc(x0+x1)

        # Residual
        # x = in1 - x
        return x
  
