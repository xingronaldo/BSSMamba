import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from .backbone.mobilenetv2 import mobilenet_v2
from .block.fpn import FPN
from .block.encoder import Encoder
from .block.decoder import Decoder
from .block.heads import GatedResidualUpHead
from .util import init_method


def get_backbone(backbone_name):
    if backbone_name == 'mobilenetv2':
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
    else:
        raise NotImplementedError("BACKBONE [%s] is not implemented!\n" % backbone_name)
    return backbone


def get_fpn(fpn_name, in_channels, out_channels):
    if fpn_name == 'fpn':
        fpn = FPN(in_channels, out_channels)
    else:
        raise NotImplementedError("FPN [%s] is not implemented!\n" % fpn_name)
    return fpn


class Detector(nn.Module):
    def __init__(self, channels=128, encoder_depth=2, decoder_depth=2, neigh_size=5, margin=0.5, init_type='normal'):
        super().__init__()
        self.neigh_size = neigh_size
        self.margin = margin
        self.backbone = get_backbone(backbone_name='mobilenetv2')
        self.fpn = get_fpn(fpn_name='fpn', in_channels=self.backbone.channels[-4:], out_channels=channels)
        self.encoder = Encoder(channels, depth=encoder_depth)
        self.decoder = Decoder(channels, depth=decoder_depth)

        self.project = nn.Sequential(nn.Conv2d(channels, channels//2, 1, padding=0),
                                     nn.ReLU6())
        self.classifier = nn.Conv2d(channels//2, 2, 1, padding=0)
        self.ds_head = nn.Sequential(nn.Conv2d(channels, channels//2, 1, padding=0),
                                      nn.ReLU6(),
                                      nn.Conv2d(channels//2, 2, 1, padding=0))
        self.head = GatedResidualUpHead(channels, 2)
        #init_method(MODULE_NAMES, init_type=init_type)

    def forward(self, x1, x2, label=None, boundary_mask=None, boundary_label=None):
        ### Extract CNN features
        _, t1_c2, t1_c3, t1_c4, t1_c5 = self.backbone.forward(x1)
        _, t2_c2, t2_c3, t2_c4, t2_c5 = self.backbone.forward(x2)
        fea1 = self.fpn([t1_c2, t1_c3, t1_c4, t1_c5])
        fea2 = self.fpn([t2_c2, t2_c3, t2_c4, t2_c5])

        en_fea = self.encoder(fea1, fea2)
        de_fea = self.decoder(en_fea)
        pred = self.head(de_fea)

        if label is None:  ### if not is_training
            return pred

        else:  ### if is_training
            label = F.interpolate(label.unsqueeze(1).float(), size=en_fea.size()[2:], mode='nearest').squeeze(1).long()
            boundary_mask = F.interpolate(boundary_mask.unsqueeze(1).float(), size=en_fea.size()[2:],
                                          mode='nearest').squeeze(1).long()
            boundary_label = F.interpolate(boundary_label.unsqueeze(1).float(), size=en_fea.size()[2:],
                                          mode='nearest').squeeze(1).long()

            ds_pred = self.ds_head(en_fea)
            ds_loss = F.cross_entropy(ds_pred, label)

            de_fea_prj = self.project(de_fea)
            one_hot_label = torch.eye(2, device=de_fea_prj.device)[label].permute(0, 3, 1, 2).type_as(de_fea_prj)
            boundary_coh_loss = self.boundary_coherency_loss(de_fea_prj, one_hot_label,
                                                                    boundary_mask, pred)
            pred_boundary = self.classifier(de_fea_prj)
            boundary_cls_loss = F.cross_entropy(pred_boundary, boundary_label, ignore_index=255)

            return pred, ds_loss, boundary_coh_loss, boundary_cls_loss

    def get_neigh(self, input):
        if self.neigh_size % 2 == 0:
            raise ValueError("K must be an odd number to have a center element.")
        kernel = np.ones((self.neigh_size, self.neigh_size), dtype='float32')
        kernel[self.neigh_size // 2, self.neigh_size // 2] = 0.0
        kernel = kernel.reshape((1, 1, self.neigh_size, self.neigh_size))
        kernel = np.repeat(kernel, input.size()[1], axis=0)
        kernel = torch.from_numpy(kernel).to(input.device)
        out = F.conv2d(input, weight=kernel, padding=self.neigh_size // 2, groups=input.size()[1])
        return out

    def boundary_coherency_loss(self, fea, one_hot_label, boundary_mask, pred):
        shown_class = list(torch.argmax(one_hot_label.detach(), dim=1).unique())
        pred = torch.argmax(pred.detach(), dim=1)
        pred = F.interpolate(pred.unsqueeze(1).float(), size=fea.size()[2:], mode='nearest').squeeze(1).long()
        one_hot_pred = torch.eye(2, device=fea.device)[pred].permute(0, 3, 1, 2).type_as(fea)

        loss = torch.tensor(0.0, device=fea.device)
        num = len(shown_class)

        class0_mask, class1_mask = one_hot_label[:, 0, :, :], one_hot_label[:, 1, :, :]
        pred0_mask, pred1_mask = one_hot_pred[:, 0, :, :], one_hot_pred[:, 1, :, :]

        neigh_fea_class0 = self.get_neigh(fea * class0_mask.unsqueeze(1))
        neigh_fea_class1 = self.get_neigh(fea * class1_mask.unsqueeze(1))
        correct_neigh_fea_class0 = self.get_neigh(fea * (class0_mask * pred0_mask).unsqueeze(1))
        correct_neigh_fea_class1 = self.get_neigh(fea * (class1_mask * pred1_mask).unsqueeze(1))
        num_neigh_class0 = self.get_neigh(class0_mask.unsqueeze(1).float())
        num_neigh_class1 = self.get_neigh(class1_mask.unsqueeze(1).float())
        num_correct_neigh_class0 = self.get_neigh((class0_mask * pred0_mask).unsqueeze(1).float())
        num_correct_neigh_class1 = self.get_neigh((class1_mask * pred1_mask).unsqueeze(1).float())

        mean_neigh_fea_class0 = neigh_fea_class0 / (num_neigh_class0 + 1e-6)
        mean_correct_fea_class0 = correct_neigh_fea_class0 / (num_correct_neigh_class0 + 1e-6)
        mean_neigh_fea_class1 = neigh_fea_class1 / (num_neigh_class1 + 1e-6)
        mean_correct_fea_class1 = correct_neigh_fea_class1 / (num_correct_neigh_class1 + 1e-6)
        mean_neigh_fea_class0 = mean_neigh_fea_class0.permute(0, 2, 3, 1)
        mean_correct_fea_class0 = mean_correct_fea_class0.permute(0, 2, 3, 1)
        mean_neigh_fea_class1 = mean_neigh_fea_class1.permute(0, 2, 3, 1)
        mean_correct_fea_class1 = mean_correct_fea_class1.permute(0, 2, 3, 1)

        """Select pixel for calculating loss"""
        pixel_mask_class0_pos = (num_correct_neigh_class0.squeeze(1) >= 1) * boundary_mask.bool()
        pixel_mask_class1_pos = (num_correct_neigh_class1.squeeze(1) >= 1) * boundary_mask.bool()
        pixel_mask_class0_neg = (num_neigh_class0.squeeze(1) >= 1) * pixel_mask_class1_pos
        pixel_mask_class1_neg = (num_neigh_class1.squeeze(1) >= 1) * pixel_mask_class0_pos

        """Positive part"""
        mean_neigh_fea_class0_pos = mean_neigh_fea_class0[pixel_mask_class0_pos]
        mean_correct_fea_class0_pos = mean_correct_fea_class0[pixel_mask_class0_pos]
        mean_neigh_fea_class1_pos = mean_neigh_fea_class1[pixel_mask_class1_pos]
        mean_correct_fea_class1_pos = mean_correct_fea_class1[pixel_mask_class1_pos]

        simi_label_class0_pos = torch.ones(mean_neigh_fea_class0_pos.size()[0], device=mean_neigh_fea_class0_pos.device)
        simi_label_class1_pos = torch.ones(mean_neigh_fea_class1_pos.size()[0], device=mean_neigh_fea_class1_pos.device)
        loss += F.cosine_embedding_loss(mean_neigh_fea_class0_pos, mean_correct_fea_class0_pos, simi_label_class0_pos) + \
                F.cosine_embedding_loss(mean_neigh_fea_class1_pos, mean_correct_fea_class1_pos, simi_label_class1_pos)

        """Negative part"""
        mean_neigh_fea_class0_neg = mean_neigh_fea_class0[pixel_mask_class0_neg]
        mean_correct_fea_class1_neg = mean_correct_fea_class1[pixel_mask_class0_neg]
        mean_neigh_fea_class1_neg = mean_neigh_fea_class1[pixel_mask_class1_neg]
        mean_correct_fea_class0_neg = mean_correct_fea_class0[pixel_mask_class1_neg]
        simi_label_class0_neg = -1 * torch.ones(mean_neigh_fea_class0_neg.size()[0],
                                                device=mean_neigh_fea_class0_neg.device)
        simi_label_class1_neg = -1 * torch.ones(mean_neigh_fea_class1_neg.size()[0],
                                                device=mean_neigh_fea_class1_neg.device)

        loss += F.cosine_embedding_loss(mean_neigh_fea_class0_neg, mean_correct_fea_class1_neg, simi_label_class0_neg,
                                        margin=self.margin) + \
                F.cosine_embedding_loss(mean_neigh_fea_class1_neg, mean_correct_fea_class0_neg, simi_label_class1_neg,
                                        margin=self.margin)
        loss = loss / num
        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=fea.device)
        return loss


