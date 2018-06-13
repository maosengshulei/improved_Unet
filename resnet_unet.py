import torch
import torch.nn as nn
import torchvision
import resnet
import resnext
from resnet import Bottleneck
model_urls = {
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
}


class SegmentationModule(nn.Module):
    def __init__(self, net_enc, net_dec,deep_sup_factor):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.deep_sup_factor=deep_sup_factor

    def forward(self,img):
        if self.deep_sup_factor>0:
            pred,pred_sup=self.decoder(self.encoder(img))
            return pred,pred_sup
        else:
            pred=self.decoder(self.encoder(img))
            return pred

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class DecoderBlockV2(nn.Module):
    def __init__(self,
                 in_channels,
                 middle_channels,
                 out_channels,
                 is_deconv=False):

        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                conv3x3_bn_relu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                conv3x3_bn_relu(in_channels, middle_channels),
                conv3x3_bn_relu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class ModelBuilder():
    # custom weights initialization
    '''
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)
    '''

    def build_encoder(self, arch='resnet50_dilated8', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        if arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34_dilated8':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet34_dilated16':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50_dilated8':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet50_dilated16':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnext50':
            orig_resnext = resnext.__dict__['resnext50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        else:
            raise Exception('Architecture undefined!')

        # net_encoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    def build_decoder(self, arch='ppm_bilinear_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        if arch == 'c1_bilinear_deepsup':
            net_decoder = C1BilinearDeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1_bilinear':
            net_decoder = C1Bilinear(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c2_bilinear':
            net_decoder = C2Bilinear(
                num_class=num_class,
                num_filters=32,
                is_deconv=False)
        elif arch == 'c2_bilinearwithastorous16':
            net_decoder = C2Bilinearwithastorous16(
                num_class=num_class,
                num_filters=32,
                is_deconv=False)
        elif arch == 'c2_bilinearwithastorous8':
            net_decoder = C2Bilinearwithastorous8(
                num_class=num_class,
                num_filters=32,
                is_deconv=False)

        elif arch == 'ppm_bilinear':
            net_decoder = PPMBilinear(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_bilinear_deepsup':
            net_decoder = PPMBilinearDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        elif arch == 'deep_resunet':
            net_decoder = deep_residual_unet(
                num_class=num_class
                )
        elif arch == 'recurrent_unet':
            net_decoder = RCL_Unet(num_class=num_class)
        else:
            raise Exception('Architecture undefined!')

        # net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder



class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x):
        conv_out=[]
        x = self.relu1(self.bn1(self.conv1(x)))

        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        conv_out.append(x)
        x = self.layer1(x);conv_out.append(x)
        x = self.layer2(x);conv_out.append(x)
        x = self.layer3(x);conv_out.append(x)
        x = self.layer4(x);conv_out.append(x)
        return conv_out


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);


        return conv_out

class C1BilinearDeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1BilinearDeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)
        '''
        if self.use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x
        '''
        x=nn.functional.upsample(x,scale_factor=16,mode='bilinear')
        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        _=nn.functional.upsample(_,scale_factor=16,mode='bilinear')

        return (x, _)

class C1Bilinear(nn.Module):
    #dialated8 and upsample8
    def __init__(self, num_class=1, fc_dim=2048, use_softmax=False):
        super(C1Bilinear, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        '''
        if self.use_softmax: # is True during inference
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        '''
        x = nn.functional.upsample(x, scale_factor=8, mode='bilinear')
        return x

class C2Bilinear(nn.Module):
    #no dialation and u-net decoder
    def __init__(self,num_class=1,num_filters=32,is_deconv=False):
        super(C2Bilinear,self).__init__()
        self.center = DecoderBlockV2(2048, num_filters * 8 * 2, num_filters * 8,is_deconv)
        self.pool = nn.MaxPool2d(2, 2)
        self.dec5 = DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.cbr = conv3x3_bn_relu(num_filters, num_filters, 1)

        # last conv
        self.conv_last = nn.Conv2d(num_filters,num_class, 1, 1, 0)

    def forward(self,conv_out):
        center=self.center(self.pool(conv_out[-1]))
        dec5=self.dec5(torch.cat([center,conv_out[-1]],1))
        dec4 = self.dec4(torch.cat([dec5, conv_out[-2]], 1))
        dec3 = self.dec3(torch.cat([dec4, conv_out[-3]], 1))
        dec2 = self.dec2(torch.cat([dec3, conv_out[-4]], 1))
        dec1 = self.dec1(dec2)
        x=self.cbr(dec1)
        x = self.conv_last(x)
        return x


class C2Bilinearwithastorous16(nn.Module):
    #dialated16 and u-net decoder
    def __init__(self,num_class=1,num_filters=32,is_deconv=False):
        super(C2Bilinearwithastorous16,self).__init__()
        self.center = DecoderBlockV2(2048, num_filters * 8 * 2, num_filters * 8,is_deconv)
        self.pool = nn.MaxPool2d(2, 2)
        self.dec4 = DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2 , is_deconv)
        self.dec2 = DecoderBlockV2(256 + num_filters * 2 , num_filters * 2 * 2, num_filters*2*2, is_deconv)
        #self.dec2 = DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.cbr = conv3x3_bn_relu(num_filters, num_filters, 1)

        # last conv
        self.conv_last = nn.Conv2d(num_filters,num_class, 1, 1, 0)

    def forward(self,conv_out):
        center=self.center(self.pool(conv_out[-1]))
        dec4=self.dec4(torch.cat([center,conv_out[-1]],1))
        dec3=self.dec3(torch.cat([dec4,conv_out[-3]],1))
        dec2 = self.dec2(torch.cat([dec3, conv_out[-4]], 1))
        dec1 = self.dec1(dec2)
        #dec2 = self.dec2(torch.cat([dec3, conv_out[-4]], 1))
        #dec1 = self.dec1(dec2)
        x=self.cbr(dec1)
        x = self.conv_last(x)
        return x

class C2Bilinearwithastorous8(nn.Module):
    #dialated8 and u-net decoder
    def __init__(self,num_class=1,num_filters=32,is_deconv=False):
        super(C2Bilinearwithastorous8,self).__init__()
        self.center = DecoderBlockV2(2048, num_filters * 8 * 2, num_filters * 8,is_deconv)
        self.pool = nn.MaxPool2d(2, 2)
        self.dec3 = DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        #self.dec3 = DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2 , is_deconv)
        self.dec2 = DecoderBlockV2(256 + num_filters * 8 , num_filters * 4 * 2, num_filters*2*2, is_deconv)
        #self.dec2 = DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.cbr = conv3x3_bn_relu(num_filters, num_filters, 1)

        # last conv
        self.conv_last = nn.Conv2d(num_filters,num_class, 1, 1, 0)

    def forward(self,conv_out):
        center=self.center(self.pool(conv_out[-1]))
        dec3=self.dec3(torch.cat([center,conv_out[-1]],1))
        #dec3=self.dec3(torch.cat([dec4,conv_out[-3]],1))
        dec2 = self.dec2(torch.cat([dec3, conv_out[-4]], 1))
        dec1 = self.dec1(dec2)
        #dec2 = self.dec2(torch.cat([dec3, conv_out[-4]], 1))
        #dec1 = self.dec1(dec2)
        x=self.cbr(dec1)
        x = self.conv_last(x)
        return x

class PPMBilinear(nn.Module):
    def __init__(self, num_class=1,fc_dim=2048,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinear, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear'))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)


        x = nn.functional.upsample(
            x, scale_factor=32, mode='bilinear')

        return x

class PPMBilinearDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinearDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear'))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)
        x=nn.functional.upsample(x,scale_factor=16,mode='bilinear')
        '''
        if self.use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x
        '''
        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)
        _=nn.functional.upsample(_,scale_factor=16,mode='bilinear')


        return (x, _)

class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256,512,1024,2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear')))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(1,len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i-1](conv_x) # lateral branch

            f = nn.functional.upsample(
                f, size=conv_x.size()[2:], mode='bilinear') # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i-1](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.upsample(
                fpn_feature_list[i],
                output_size,
                mode='bilinear'))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        x=nn.functional.upsample(
            x,scale_factor=4,mode='bilinear')
        '''
        if self.use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)
        '''
        return x


class deep_residual_unet(nn.Module):

    def __init__(self,num_class=1):
        self.inplanes = 2048
        super(deep_residual_unet,self).__init__()


        self.center=self._make_layer(Bottleneck,2048,32*8,1,stride=2)
        self.dec5=self._make_layer(Bottleneck,2048+32*8*4,32*8,1,stride=1)
        self.dec4=self._make_layer(Bottleneck,1024+32*8*4,32*4,1,stride=1)
        self.dec3=self._make_layer(Bottleneck,512+32*4*4,32*2,1,stride=1)
        self.dec2=self._make_layer(Bottleneck,256+32*2*4,32,1,stride=1)
        self.dec1=self._make_layer(Bottleneck,32*4,16,1,stride=1)
        self.upsample2x=nn.Upsample(scale_factor=2, mode='bilinear')
        self.cbr = conv3x3_bn_relu(64,64, 1)

    # last conv
        self.conv_last = nn.Conv2d(64,num_class, 1, 1, 0)

    def _make_layer(self, block, in_planes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(in_planes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self,conv_out):
        center=self.upsample2x(self.center(conv_out[-1]))
        dec5=self.upsample2x(self.dec5(torch.cat([conv_out[-1],center],1)))
        dec4=self.upsample2x(self.dec4(torch.cat([conv_out[-2],dec5],1)))
        dec3=self.upsample2x(self.dec3(torch.cat([conv_out[-3],dec4],1)))
        dec2=self.upsample2x(self.dec2(torch.cat([conv_out[-4],dec3],1)))
        dec1=self.upsample2x(self.dec1(dec2))
        x=self.cbr(dec1)
        x = self.conv_last(x)
        return x


class RCLblock(nn.Module):
    def __init__(self,inplanes,planes):
        super(RCLblock,self).__init__()
        self.conv1=nn.Sequential(conv3x3(inplanes,planes),nn.BatchNorm2d(planes))
        self.rcl=nn.Sequential(conv3x3(planes,planes),nn.BatchNorm2d(planes))
        self.bn=nn.BatchNorm2d(planes)
        self.rl=nn.ReLU(inplace=True)

    def forward(self,input_map):
        conv1=self.conv1(input_map)
        conv2=self.rcl(conv1)
        conv2 += conv1
        conv2=self.bn(conv2)
        conv3=self.rcl(conv2)
        conv3 += conv1
        conv3=self.bn(conv3)
        x=self.rl(conv3)
        return x

class RCL_Unet(nn.Module):
    def __init__(self,num_class=1,planes=32):
        super(RCL_Unet,self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.center=nn.Sequential(
            RCLblock(2048,planes*8),
            nn.MaxPool2d(2,2)
            )
        self.dec5=RCLblock(2048+planes*8,planes*8)
        self.dec4=RCLblock(1024+planes*8,planes*4)
        self.dec3=RCLblock(512+planes*4,planes*4)
        self.dec2=RCLblock(256+planes*4,planes*2)
        self.dec1=RCLblock(planes*2,planes)
        self.upsample2x=nn.Upsample(scale_factor=2, mode='bilinear')

        self.cbr = conv3x3_bn_relu(planes,planes, 1)

        # last conv
        self.conv_last = nn.Conv2d(planes,num_class, 1, 1, 0)

    def forward(self,conv_out):
        center=self.upsample2x(self.center(conv_out[-1]))
        dec5=self.upsample2x(self.dec5(torch.cat([conv_out[-1],center],1)))
        dec4=self.upsample2x(self.dec4(torch.cat([conv_out[-2],dec5],1)))
        dec3=self.upsample2x(self.dec3(torch.cat([conv_out[-3],dec4],1)))
        dec2=self.upsample2x(self.dec2(torch.cat([conv_out[-4],dec3],1)))
        dec1=self.upsample2x(self.dec1(dec2))
        x=self.cbr(dec1)
        x = self.conv_last(x)
        return x




class Loss:
    def __init__(self, dice_weight=1):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.dice_weight:
            eps = 1e-15

            dice_target = (targets == 1).float()
            dice_output = outputs
            dice_output=F.sigmoid(dice_output)
            intersection = (dice_output * dice_target).sum()
            union = dice_output.sum() + dice_target.sum() + eps

            loss -= torch.log(2 * intersection / union)

        return loss
