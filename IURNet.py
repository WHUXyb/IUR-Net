import sys
sys.path.append('..')  # 将父级目录添加到导入搜索路径中 改  board and google库
from models.ResNet import ResNet50
import numpy as np
from models.attention_block import *
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


class IURNet(nn.Module):
    def __init__(self, backbone='ResNet50',
                 in_channels=4,
                 num_classes=1,
                 with_aux=True,
                 with_mode='Ms-TripletAttention'):
                 # with_mode=None):
        super(IURNet, self).__init__()
        self.input_mode = 0 if in_channels == 4 else 1  # 0: concatenate, 1:only image
        self.with_aux = with_aux
        stage_channels = []
        self.downsample4 = True
        self.backbone_name = backbone
        self.config_vit = None
        if backbone == 'ResNet50':
            self.backbone = HRNetV2(in_channels=in_channels, hr_cfg='w48')
            stage_channels = self.backbone.stage_channels

        elif backbone == 'transunet':
            from models.vit_seg_modeling import get_transunet_backbone
            self.backbone, config_vit = get_transunet_backbone(in_channel=in_channels, img_size=512)
            self.config_vit = config_vit
            # stage_channels = [16, 16, 16, 16]
            stage_channels = [16, 64, 128, 256]
            self.downsample4 = False

        if self.with_aux:
            last_inp_channels = int(np.sum(stage_channels))
            self.fusion = nn.Sequential(
                nn.Conv2d(last_inp_channels, last_inp_channels, 5, 1, 2),
                nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
            self.seg_head = nn.Sequential(
                nn.Conv2d(last_inp_channels, num_classes, 1, 1, 0),
                nn.Sigmoid()
            )

        out_channels = 64
        if self.downsample4:
            self.deconv = nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False)

        self.feature_block = CSAMBlock(out_channels, stage_channels, attention=with_mode)

        self.pre_class = nn.Sequential(
            nn.Conv2d(out_channels, 2, 1, 1, 0),
            nn.Sigmoid()
        )
        self.init_weights()

    def base_forward(self, x, check_map):
        _, c, m, n = x.shape
        if self.input_mode == 0:
            y_list = self.backbone(torch.cat((x, check_map), dim=1))
        else:
            y_list = self.backbone(x)

        if self.with_aux:
            x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
            x_list = [F.upsample(y, size=(x0_h, x0_w), mode='bilinear', align_corners=True) for y in y_list]
            x = self.fusion(torch.cat(x_list, 1))
            binary = self.seg_head(x)
            binary = F.upsample(input=binary, size=(m, n), mode='bilinear', align_corners=True)

        fea_re = self.feature_block(y_list, check_map)  #重点

        if self.downsample4:
            fea_re = self.deconv(fea_re)

        building_res = self.pre_class(fea_re)

        if self.downsample4:
            building_res = F.upsample(input=building_res, size=(m, n), mode='bilinear', align_corners=True)
        if self.with_aux:
            res = [building_res, binary]
        else:
            res = [building_res, None]
            # # Replace None with a zero tensor or some placeholder tensor
            # placeholder_tensor = torch.zeros_like(building_res)
            # res = [building_res, placeholder_tensor]
        
        # return res
        return res, fea_re  #修改

    def forward(self, x, check_map):
        return self.base_forward(x, check_map)

    def init_weights(self, pretrained='', model_pretrained=''):
        # logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.backbone_name == 'transunet':
            if os.path.exists(self.config_vit.pretrained_path):
                print('load transunet preweights successfully!')
                self.backbone.load_from(weights=np.load(self.config_vit.pretrained_path))
            else:
                print('load transunet preweights failed!')

        if os.path.isfile(pretrained):
            pre_weight = torch.load(pretrained)
            # logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.backbone.state_dict()
            pretrained_dict = {}
            old_conv1_weight = pre_weight['conv1.weight']
            channel = model_dict['conv1.weight'].size()[1]
            num = 0
            for i in range(0, channel):
                model_dict['conv1.weight'][:, i, :, :] = old_conv1_weight[:, i % 3, :, :]
                num += 1

            for k, v in pre_weight.items():
                if k in model_dict.keys() and k != 'conv1.weight':
                    pretrained_dict[k] = v
                    num += 1
            model_dict.update(pretrained_dict)
            print(f'load-weight:{num}/{len(pre_weight.keys())}')
            self.backbone.load_state_dict(model_dict)

        if os.path.isfile(model_pretrained):
            pre_weight = torch.load(model_pretrained, map_location='cpu')
            model_dict = self.state_dict()
            num = 0
            for k, v in pre_weight.items():
                if 'model.' in k: k = k[6:]
                if k in model_dict.keys():
                    model_dict[k] = v
                    num += 1
            print(f'model-load-weight:{num}/{len(pre_weight.keys())}')
            self.load_state_dict(model_dict)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    from thop import profile
    inputs = torch.randn(1, 3, 512, 512).cuda()
    check_map = torch.randn(1, 1, 512, 512).cuda()
    model = IURNet(backbone = 'transunet', in_channels=4, with_mode='scam', with_aux=False).cuda()  #with_mode='scam'with_mode='scam'
    macs, params = profile(model, inputs=(inputs,check_map,))
    name = 'dis'
    print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
    print("---|---|---")
    print("%s | %.2f | %.2f" % (name, params / (1000 ** 2), macs / (1000 ** 3)))
