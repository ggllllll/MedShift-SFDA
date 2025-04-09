from torch import nn
import torch
from models.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from tool import pyutils
import sys
import torch.sparse as sparse

class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class UNet(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False, image_res=512, radius=4):
        super().__init__()
        cut, lr_cut = [8, 6]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        layers = list(base_model(pretrained=pretrained).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.Dropout1 = nn.Dropout(0.1)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)


        feature_num = 512
        self.aff_cup = torch.nn.Conv2d(feature_num, feature_num, 1, bias=False)
        self.aff_disc = torch.nn.Conv2d(feature_num, feature_num, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.aff_cup.weight, gain=4)
        torch.nn.init.xavier_uniform_(self.aff_disc.weight, gain=4)
        self.bn_cup = SynchronizedBatchNorm2d(feature_num)
        self.bn_disc = SynchronizedBatchNorm2d(feature_num)
        self.bn_cup.weight.data.fill_(1)
        self.bn_cup.bias.data.zero_()
        self.bn_disc.weight.data.fill_(1)
        self.bn_disc.bias.data.zero_()

        self.from_scratch_layers = [self.aff_cup, self.aff_disc, self.bn_cup, self.bn_disc]

        self.predefined_featuresize = 32
        self.ind_from, self.ind_to = pyutils.get_indices_of_pairs(radius=radius, size=(self.predefined_featuresize, self.predefined_featuresize))
        self.ind_from = torch.from_numpy(self.ind_from);
        self.ind_to = torch.from_numpy(self.ind_to)

    def forward(self, x, to_dense=False, rfeat=True):
        x = F.relu(self.rn(x))
        # x = self.Dropout1(x)
        feature = x #torch.Size([8, 256, 32, 32])
        x = self.up1(x, self.sfs[3].features)
        feature = F.interpolate(feature, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        fea = x
        output = self.up5(x)
        output = output[:, [1, 0], :, :]

        # x = F.relu(self.rn(x))
        # x = self.up1(self.Dropout1(x), self.Dropout1(self.sfs[3].features))
        # feature = x #torch.Size([8, 256, 32, 32])
        # x = self.up2(self.Dropout1(x), self.Dropout1(self.sfs[2].features))
        # x = self.up3(self.Dropout1(x), self.Dropout1(self.sfs[1].features))
        # x = self.up4(self.Dropout1(x), self.Dropout1(self.sfs[0].features))
        # x = self.Dropout1(x)
        # fea = x
        # output = self.up5(x)

   
        f_cup = F.relu(self.bn_cup(self.aff_cup(feature)))  ###bn
        f_disc = F.relu(self.bn_disc(self.aff_disc(feature)))

        if f_cup.size(2) == self.predefined_featuresize and f_cup.size(3) == self.predefined_featuresize:
            ind_from = self.ind_from
            ind_to = self.ind_to
        else:
            print('featuresize error')
            sys.exit()

        f_cup = f_cup.view(f_cup.size(0), f_cup.size(1), -1)

        ff = torch.index_select(f_cup, dim=2, index=ind_from.cuda(non_blocking=True))
        ft = torch.index_select(f_cup, dim=2, index=ind_to.cuda(non_blocking=True))

        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))

        aff_cup = torch.exp(-torch.mean(torch.abs(ft - ff), dim=1))

        if to_dense:
            aff_cup = aff_cup.view(-1).cpu()

            ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
            indices = torch.stack([ind_from_exp, ind_to])
            indices_tp = torch.stack([ind_to, ind_from_exp])

            area = f_cup.size(2)
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

            aff_cup = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                         torch.cat([aff_cup, torch.ones([area]), aff_cup])).to_dense().cuda()

        f_disc = f_disc.view(f_disc.size(0), f_disc.size(1), -1)

        ff = torch.index_select(f_disc, dim=2, index=ind_from.cuda(non_blocking=True))
        ft = torch.index_select(f_disc, dim=2, index=ind_to.cuda(non_blocking=True))

        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))

        aff_disc = torch.exp(-torch.mean(torch.abs(ft - ff), dim=1))

        if to_dense:
            aff_disc = aff_disc.view(-1).cpu()

            ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
            indices = torch.stack([ind_from_exp, ind_to])
            indices_tp = torch.stack([ind_to, ind_from_exp])

            area = f_disc.size(2)
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

            aff_disc = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                          torch.cat([aff_disc, torch.ones([area]), aff_disc])).to_dense().cuda()

        if not rfeat:
            return output
        else:
            return output, None, fea, aff_cup, aff_disc

    def close(self):
        for sf in self.sfs: sf.remove()

    def get_scratch_parameters(self):
        groups = []

        for param in self.parameters():
            param.requires_grad = False

        for m in self.modules():

            if m in self.from_scratch_layers:

                groups.append(m.weight)
                m.weight.requires_grad = True
                if isinstance(m, SynchronizedBatchNorm2d):
                    groups.append(m.bias)
                    m.bias.requires_grad = True
                # groups.append(m.bias)###
        return groups
