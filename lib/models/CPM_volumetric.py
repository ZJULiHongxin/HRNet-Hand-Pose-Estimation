import torch.nn as nn
import torch.nn.functional as F
import torch

BN_MOMENTUM = 0.1

class GlobalAveragePoolingHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)

        batch_size, n_channels = x.shape[:2]
        x = x.view((batch_size, n_channels, -1))
        x = x.mean(dim=-1)

        out = self.head(x)

        return out


class CPM(nn.Module):
    def __init__(self, cfg):
        super(CPM, self).__init__()
        self.hm_size = cfg.MODEL.HEATMAP_SIZE[0]
        self.k = cfg.DATASET.NUM_JOINTS
        self.pool_center = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)
        self.conv1_stage1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.pool1_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_stage1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv5_stage1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv6_stage1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv7_stage1 = nn.Conv2d(512, self.k + 1, kernel_size=1)

        self.conv1_stage2 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.pool1_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_stage2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage2 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mconv2_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage2 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage3 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage3 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mconv2_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage3 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage4 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage4 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mconv2_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage4 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage5 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage5 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mconv2_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage5 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage5 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage6 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage6 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mconv2_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage6 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage6 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        if cfg.MODEL.VOL_CONFIDENCES:
            self.vol_confidences = GlobalAveragePoolingHead(128, 32)

    def _stage1(self, image):

        x = self.pool1_stage1(F.relu(self.conv1_stage1(image)))
        x = self.pool2_stage1(F.relu(self.conv2_stage1(x)))
        x = self.pool3_stage1(F.relu(self.conv3_stage1(x)))
        x = F.relu(self.conv4_stage1(x))
        x = F.relu(self.conv5_stage1(x))
        x = F.relu(self.conv6_stage1(x))
        x = self.conv7_stage1(x)

        return x

    def _middle(self, image):

        x = self.pool1_stage2(F.relu(self.conv1_stage2(image)))
        x = self.pool2_stage2(F.relu(self.conv2_stage2(x)))
        x = self.pool3_stage2(F.relu(self.conv3_stage2(x)))

        return x

    def _stage2(self, pool3_stage2_map, conv7_stage1_map, pool_center_map):
        
        x = F.relu(self.conv4_stage2(pool3_stage2_map))
        x = torch.cat([x, conv7_stage1_map, pool_center_map], dim=1)
        x = F.relu(self.Mconv1_stage2(x))
        x = F.relu(self.Mconv2_stage2(x))
        x = F.relu(self.Mconv3_stage2(x))
        x = F.relu(self.Mconv4_stage2(x))
        x = self.Mconv5_stage2(x)

        return x

    def _stage3(self, pool3_stage2_map, Mconv5_stage2_map, pool_center_map):

        x = F.relu(self.conv1_stage3(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage2_map, pool_center_map], dim=1)
        x = F.relu(self.Mconv1_stage3(x))
        x = F.relu(self.Mconv2_stage3(x))
        x = F.relu(self.Mconv3_stage3(x))
        x = F.relu(self.Mconv4_stage3(x))
        x = self.Mconv5_stage3(x)

        return x

    def _stage4(self, pool3_stage2_map, Mconv5_stage3_map, pool_center_map):

        x = F.relu(self.conv1_stage4(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage3_map, pool_center_map], dim=1)
        x = F.relu(self.Mconv1_stage4(x))
        x = F.relu(self.Mconv2_stage4(x))
        x = F.relu(self.Mconv3_stage4(x))
        x = F.relu(self.Mconv4_stage4(x))
        x = self.Mconv5_stage4(x)

        return x

    def _stage5(self, pool3_stage2_map, Mconv5_stage4_map, pool_center_map):

        x = F.relu(self.conv1_stage5(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage4_map, pool_center_map], dim=1)
        x = F.relu(self.Mconv1_stage5(x))
        x = F.relu(self.Mconv2_stage5(x))
        x = F.relu(self.Mconv3_stage5(x))
        x = F.relu(self.Mconv4_stage5(x))
        x = self.Mconv5_stage5(x)

        return x

    def _stage6(self, pool3_stage2_map, Mconv5_stage5_map, pool_center_map):
        
        x = F.relu(self.conv1_stage6(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage5_map, pool_center_map], dim=1) # 4 x 55 x 32 x 32
        
        x = F.relu(self.Mconv1_stage6(x))
        x = F.relu(self.Mconv2_stage6(x)) # 128

        x = F.relu(self.Mconv3_stage6(x))
        x = F.relu(self.Mconv4_stage6(x)) # 128
        inter_feat = x
        x = self.Mconv5_stage6(x)
        return x, inter_feat



    def forward(self, image, center_map=torch.ones((1,1,256,256))):


        pool_center_map = self.pool_center(center_map)

        conv7_stage1_map = self._stage1(image)

        pool3_stage2_map = self._middle(image)

        Mconv5_stage2_map = self._stage2(pool3_stage2_map, conv7_stage1_map,
                                         pool_center_map)
        Mconv5_stage3_map = self._stage3(pool3_stage2_map, Mconv5_stage2_map,
                                         pool_center_map)
        Mconv5_stage4_map = self._stage4(pool3_stage2_map, Mconv5_stage3_map,
                                         pool_center_map)
        Mconv5_stage5_map = self._stage5(pool3_stage2_map, Mconv5_stage4_map,
                                         pool_center_map)

        Mconv5_stage6_map, inter_feat = self._stage6(pool3_stage2_map, Mconv5_stage5_map,
                                         pool_center_map)

        vol_confidences = None
        if hasattr(self, 'vol_confidences'):
            inter_feat = F.interpolate(inter_feat, size=(self.hm_size, self.hm_size), mode='bilinear', align_corners=True) # b x 55 x 32 x 32 -> b x 55 x 64 x 64
            vol_confidences = self.vol_confidences(inter_feat)   # size b x N_joints
        
        Mconv5_stage6_map = F.interpolate(Mconv5_stage6_map, size=(self.hm_size, self.hm_size), mode='bilinear', align_corners=True) # 32x32 -> 64x64

        return conv7_stage1_map, Mconv5_stage2_map, Mconv5_stage3_map, Mconv5_stage4_map, Mconv5_stage5_map, Mconv5_stage6_map, inter_feat, vol_confidences

def get_pose_net(cfg, is_train, **kwargs):
    model = CPM(cfg)

    return model