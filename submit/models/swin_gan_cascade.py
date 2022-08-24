from models.miziha import SwinT
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from models.networks import get_pad
from models.networks import ConvWithActivation
from models.networks import DeConvWithActivation


class Residual(nn.Layer):

    def __init__(self, in_channels, out_channels, same_shape=True):
        super(Residual, self).__init__()
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(in_channels, in_channels, kernel_size=3,
                               padding=1, stride=strides)
        self.conv2 = nn.Conv2D(in_channels, out_channels, kernel_size=3,
                               padding=1)
        if not same_shape:
            self.conv3 = nn.Conv2D(in_channels, out_channels, kernel_size=1,
                                   stride=strides)
        self.batch_norm2d = nn.BatchNorm2D(out_channels)

    # noinspection PyShadowingNames
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        out = self.batch_norm2d(out + x)
        return F.relu(out)
class NonLocalBlock(nn.Layer):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2D(channel, self.inter_channel, kernel_size=1, stride=1, bias_attr=False)
        self.conv_theta = nn.Conv2D(channel, self.inter_channel, kernel_size=1, stride=1, bias_attr=False)
        self.conv_g = nn.Conv2D(channel, self.inter_channel, kernel_size=1, stride=1, bias_attr=False)
        self.softmax = nn.Softmax(axis=1)
        self.conv_mask = nn.Conv2D(self.inter_channel, channel, kernel_size=1, stride=1, bias_attr=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.shape
        # 获取phi特征，维度为[N, C/2, H * W]，注意是要保留batch和通道维度的，是在HW上
        x_phi = self.conv_phi(x)
        x_phi = paddle.reshape(x_phi, (b, c, -1))
        # 获取theta特征，维度为[N, H * W, C/2]
        x_theta = self.conv_theta(x)
        x_theta = paddle.transpose(paddle.reshape(x_theta, (b, c, -1)), (0, 2, 1))
        # 获取g特征，维度为[N, H * W, C/2]
        x_g = self.conv_g(x)
        # x_g = paddle.reshape(x_g, (b, c, -1)).permute(0, 2, 1).contiguous()
        x_g = paddle.transpose(paddle.reshape(x_g, (b, c, -1)), (0, 2, 1))
        # 对phi和theta进行矩阵乘，[N, H * W, H * W]
        # print(x_theta.shape, x_phi.shape) # [1, 8192, 64] [1, 64, 8192]
        mul_theta_phi = paddle.matmul(x_theta, x_phi)
        # softmax拉到0~1之间
        # print(mul_theta_phi.shape) # [1, 8192, 8192]
        mul_theta_phi = self.softmax(mul_theta_phi)
        # 与g特征进行矩阵乘运算，[N, H * W, C/2]
        mul_theta_phi_g = paddle.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = paddle.transpose(mul_theta_phi_g, (0, 2, 1))
        mul_theta_phi_g = paddle.reshape(mul_theta_phi_g, (b, self.inter_channel, h, w))
        # 1X1卷积扩充通道数
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x # 残差连接
        return out


class AIDR(nn.Layer):
    def __init__(self, in_channels=3, out_channels=3, num_c=48):
        super(AIDR, self).__init__()
        self.en_block1 = nn.Sequential(
            nn.Conv2D(in_channels, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2))

        self.en_block2 = nn.Sequential(
            nn.Conv2D(num_c, num_c, 3, padding=1,bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2))

        self.en_block3 = nn.Sequential(
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2))

        self.en_block4 = nn.Sequential(
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2))

        self.en_block5 = nn.Sequential(
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            NonLocalBlock(num_c),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2),
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            NonLocalBlock(num_c),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block1 = nn.Sequential(
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            NonLocalBlock(num_c*2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2,mode='nearest'))

        self.de_block2 = nn.Sequential(
            nn.Conv2D(num_c*3 , num_c*2, 3, padding=1,bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2,mode='nearest'))

        self.de_block3 = nn.Sequential(
            nn.Conv2D(num_c*3, num_c*2, 3, padding=1,bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block4 = nn.Sequential(
            nn.Conv2D(num_c*3, num_c*2, 3, padding=1,bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2,mode='nearest'))

        self.de_block5 = nn.Sequential(
            nn.Conv2D(num_c*2 + in_channels, 64, 3,padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(64, 32, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(32, out_channels, 3, padding=1, bias_attr=True))

    def forward(self, x):
        # x -> x_o_unet: h, w
        # con_x1: h/2, w/2 # [1, 32, 32, 32]
        # con_x2: h/4, w/4 # [1, 64, 16, 16]
        # con_x3: h/8, w/8 # [1, 128, 8, 8]
        # con_x4: h/16, w/16 # [1, 256, 4, 4]
        pool1 = self.en_block1(x)      # h/2, w/2
        pool2 = self.en_block2(pool1)  # h/4, w/4
        pool3 = self.en_block3(pool2)  # h/8, w/8
        pool4 = self.en_block4(pool3)  # h/16, w/16
        # print('11111111111', con_x2.shape, con_x3.shape, con_x4.shape)
        # print('11111111111', pool2.shape, pool3.shape, pool4.shape)
        upsample5 = self.en_block5(pool4)
        concat5 = paddle.concat((upsample5, pool4), axis=1)
        upsample4 = self.de_block1(concat5)
        concat4 = paddle.concat((upsample4, pool3), axis=1)
        upsample3 = self.de_block2(concat4) # h/8, w/8
        concat3 = paddle.concat((upsample3, pool2), axis=1)
        upsample2 = self.de_block3(concat3) # h/4, w/4
        concat2 = paddle.concat((upsample2, pool1), axis=1)
        upsample1 = self.de_block4(concat2) # h/2, w/2
        concat1 = paddle.concat((upsample1, x), axis=1)
        out = self.de_block5(concat1)
        return out


class STRnet2_change(nn.Layer):

    def __init__(self):
        super(STRnet2_change, self).__init__()
        self.conv1 = ConvWithActivation(3, 32, kernel_size=4, stride=2,
                                        padding=1)
        self.conva = ConvWithActivation(32, 32, kernel_size=3, stride=1,
                                        padding=1)
        self.convb = SwinT(32, 64, (256, 256), 2, 8, downsample=True)
        # self.convb = ConvWithActivation(32, 64, kernel_size=4, stride=2, padding=1)
        self.res1 = Residual(64, 64)
        self.res2 = Residual(64, 64)
        self.res3 = Residual(64, 128, same_shape=False)
        self.res4 = Residual(128, 128)
        self.res5 = Residual(128, 256, same_shape=False)
        self.res6 = Residual(256, 256)
        self.res7 = Residual(256, 512, same_shape=False)
        self.res8 = Residual(512, 512)
        self.conv2 = ConvWithActivation(512, 512, kernel_size=1)
        self.deconv1 = DeConvWithActivation(512, 256, kernel_size=3,
                                            padding=1, stride=2)
        self.deconv2 = DeConvWithActivation(256 * 2, 128, kernel_size=3,
                                            padding=1, stride=2)
        self.deconv3 = DeConvWithActivation(128 * 2, 64, kernel_size=3,
                                            padding=1, stride=2)
        self.deconv4 = DeConvWithActivation(64 * 2, 32, kernel_size=3,
                                            padding=1, stride=2)
        self.deconv5 = DeConvWithActivation(64, 3, kernel_size=3, padding=1,
                                            stride=2)
        self.lateral_connection1 = nn.Sequential(nn.Conv2D(256, 256,
                                                           kernel_size=1, padding=0, stride=1), nn.Conv2D(256, 512,
                                                                                                          kernel_size=3,
                                                                                                          padding=1,
                                                                                                          stride=1),
                                                 nn.Conv2D(512, 512,
                                                           kernel_size=3, padding=1, stride=1), nn.Conv2D(512, 256,
                                                                                                          kernel_size=1,
                                                                                                          padding=0,
                                                                                                          stride=1))
        self.lateral_connection2 = nn.Sequential(nn.Conv2D(128, 128,
                                                           kernel_size=1, padding=0, stride=1), nn.Conv2D(128, 256,
                                                                                                          kernel_size=3,
                                                                                                          padding=1,
                                                                                                          stride=1),
                                                 nn.Conv2D(256, 256,
                                                           kernel_size=3, padding=1, stride=1), nn.Conv2D(256, 128,
                                                                                                          kernel_size=1,
                                                                                                          padding=0,
                                                                                                          stride=1))
        self.lateral_connection3 = nn.Sequential(nn.Conv2D(64, 64,
                                                           kernel_size=1, padding=0, stride=1), nn.Conv2D(64, 128,
                                                                                                          kernel_size=3,
                                                                                                          padding=1,
                                                                                                          stride=1),
                                                 nn.Conv2D(128, 128,
                                                           kernel_size=3, padding=1, stride=1), nn.Conv2D(128, 64,
                                                                                                          kernel_size=1,
                                                                                                          padding=0,
                                                                                                          stride=1))
        self.lateral_connection4 = nn.Sequential(nn.Conv2D(32, 32,
                                                           kernel_size=1, padding=0, stride=1), nn.Conv2D(32, 64,
                                                                                                          kernel_size=3,
                                                                                                          padding=1,
                                                                                                          stride=1),
                                                 nn.Conv2D(64, 64,
                                                           kernel_size=3, padding=1, stride=1), nn.Conv2D(64, 32,
                                                                                                          kernel_size=1,
                                                                                                          padding=0,
                                                                                                          stride=1))
        self.conv_o1 = nn.Conv2D(64, 3, kernel_size=1)
        self.conv_o2 = nn.Conv2D(32, 3, kernel_size=1)
        self.mask_deconv_a = DeConvWithActivation(512, 256, kernel_size=3,
                                                  padding=1, stride=2)
        self.mask_conv_a = ConvWithActivation(256, 128, kernel_size=3,
                                              padding=1, stride=1)
        self.mask_deconv_b = DeConvWithActivation(256, 128, kernel_size=3,
                                                  padding=1, stride=2)
        self.mask_conv_b = ConvWithActivation(128, 64, kernel_size=3,
                                              padding=1, stride=1)
        self.mask_deconv_c = DeConvWithActivation(128, 64, kernel_size=3,
                                                  padding=1, stride=2)
        self.mask_conv_c = ConvWithActivation(64, 32, kernel_size=3,
                                              padding=1, stride=1)
        self.mask_deconv_d = DeConvWithActivation(64, 32, kernel_size=3,
                                                  padding=1, stride=2)
        self.mask_conv_d = nn.Conv2D(32, 3, kernel_size=1)
        n_in_channel = 3
        cnum = 32
        self.coarse_conva = ConvWithActivation(n_in_channel, cnum,
                                               kernel_size=5, stride=1, padding=2)
        # self.coarse_convb = ConvWithActivation(cnum, 2 * cnum, kernel_size=4, stride=2, padding=1)
        self.coarse_convb = SwinT(cnum, 2 * cnum, (512, 512), 2, 8, downsample=True)
        self.coarse_convc = ConvWithActivation(2 * cnum, 2 * cnum,
                                               kernel_size=3, stride=1, padding=1)
        self.coarse_convd = ConvWithActivation(2 * cnum, 4 * cnum,
                                               kernel_size=4, stride=2, padding=1)
        self.coarse_conve = ConvWithActivation(4 * cnum, 4 * cnum,
                                               kernel_size=3, stride=1, padding=1)
        self.coarse_convf = ConvWithActivation(4 * cnum, 4 * cnum,
                                               kernel_size=3, stride=1, padding=1)
        self.astrous_net = nn.Sequential(ConvWithActivation(4 * cnum, 4 *
                                                            cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
                                         ConvWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=4,
                                                            padding=get_pad(64, 3, 1, 4)),
                                         ConvWithActivation(4 * cnum, 4 *
                                                            cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
                                         ConvWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=16,
                                                            padding=get_pad(64, 3, 1, 16)))
        self.coarse_convk = ConvWithActivation(4 * cnum, 4 * cnum,
                                               kernel_size=3, stride=1, padding=1)
        self.coarse_convl = ConvWithActivation(4 * cnum, 4 * cnum,
                                               kernel_size=3, stride=1, padding=1)
        self.coarse_deconva = DeConvWithActivation(4 * cnum * 3, 2 * cnum,
                                                   kernel_size=3, padding=1, stride=2)
        self.coarse_convm = ConvWithActivation(2 * cnum, 2 * cnum,
                                               kernel_size=3, stride=1, padding=1)
        self.coarse_deconvb = DeConvWithActivation(2 * cnum * 3, cnum,
                                                   kernel_size=3, padding=1, stride=2)
        self.coarse_convn = nn.Sequential(ConvWithActivation(cnum, cnum //
                                                             2, kernel_size=3, stride=1, padding=1),
                                          ConvWithActivation(cnum //
                                                             2, 3, kernel_size=3, stride=1, padding=1, activation=None))
        self.c1 = nn.Conv2D(32, 64, kernel_size=1)
        self.c2 = nn.Conv2D(64, 128, kernel_size=1)
        self.sig = nn.Sigmoid()
        self.AIDR = AIDR()
    # noinspection PyShadowingNames
    def forward(self, x):
        x = self.conv1(x)
        x = self.conva(x)
        con_x1 = x
        x = self.convb(x)
        x = self.res1(x)
        con_x2 = x
        x = self.res2(x)
        x = self.res3(x)
        con_x3 = x
        x = self.res4(x)
        x = self.res5(x)
        con_x4 = x
        x = self.res6(x)
        x_mask = x
        x = self.res7(x)
        x = self.res8(x)
        x = self.conv2(x)
        x = self.deconv1(x)
        x = paddle.concat([self.lateral_connection1(con_x4), x], axis=1)
        x = self.deconv2(x)
        x = paddle.concat([self.lateral_connection2(con_x3), x], axis=1)
        x = self.deconv3(x)
        x = paddle.concat([self.lateral_connection3(con_x2), x], axis=1)
        x = self.deconv4(x)
        x = paddle.concat([self.lateral_connection4(con_x1), x], axis=1)
        x = self.deconv5(x)
        mm = self.mask_deconv_a(paddle.concat([x_mask, con_x4], axis=1))
        mm = self.mask_conv_a(mm)
        mm = self.mask_deconv_b(paddle.concat([mm, con_x3], axis=1))
        mm = self.mask_conv_b(mm)
        mm = self.mask_deconv_c(paddle.concat([mm, con_x2], axis=1))
        mm = self.mask_conv_c(mm)
        mm = self.mask_deconv_d(paddle.concat([mm, con_x1], axis=1))
        mm = self.mask_conv_d(mm)
        mm = self.sig(mm)
        x = self.coarse_conva(x)
        # print(x.shape)
        x = self.coarse_convb(x)
        x = self.coarse_convc(x)
        x_c1 = x
        x = self.coarse_convd(x)
        x = self.coarse_conve(x)
        x = self.coarse_convf(x)
        x_c2 = x
        x = self.astrous_net(x)
        x = self.coarse_convk(x)
        x = self.coarse_convl(x)
        x = self.coarse_deconva(paddle.concat([x, x_c2, self.c2(con_x2)], axis=1))
        x = self.coarse_convm(x)
        x = self.coarse_deconvb(paddle.concat([x, x_c1, self.c1(con_x1)], axis=1))
        x = self.coarse_convn(x)
        x = self.AIDR(x)
        return x, mm


if __name__ == '__main__':
    net = STRnet2_change()
    x = paddle.rand([1, 3, 64, 64])
    x, mm = net(x)
    print(x.shape, mm.shape)