import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_eco_cls_utils import PointNetEncoder, feature_transform_reguliarzer
from autoencoder.dvae import DiscreteVAEHDRm

class get_model(nn.Module):
    def __init__(self, num_token, num_class, dvae_ckpt):
        super(get_model, self).__init__()
        self.vae = DiscreteVAEHDRm(num_token).cuda()
        self.num_token = num_token
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        self.vae.load_state_dict(base_ckpt, strict=True)
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=256+64)

        self.g = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256))

        self.h = nn.Sequential(
            nn.BatchNorm1d(num_token),
            nn.Dropout(p=0.5),
            nn.Linear(num_token, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_token, bias=False))


        self.f = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_class))



    def forward(self, feature_b, neighborhood_t, center_t, gt, fine_tune=False):
        #logits_b = self.dvaehdrm.dgcnn_b(feature_b, center_b)  # B Gb Nb
        feature_t = self.vae.encoder_t(neighborhood_t)  # B Gt Ct
        logits_t = self.vae.dgcnn_t(feature_t, center_t)  # B Gt Nt
        #shuffle
        idx = torch.randperm(feature_t.shape[0])
        feature_t_shuffle = feature_t[idx, :, :].view(feature_t.size())
        logtis_t_shuffle = logits_t[idx, :, :].view(logits_t.size())

        o1 = torch.cat((feature_t, feature_b), -1)  # B G Ct+Cb
        m1 = torch.cat((feature_t_shuffle, feature_b), -1)  # B G Ct+Cb

        o2 = self.feat(o1.transpose(1,2))
        m2 = self.feat(m1.transpose(1,2))

        x = torch.cat((o2, m2), 1)

        if not fine_tune:
            o3 = self.g(o2)
            m3 = self.g(m2)

            o4 = self.h(logits_t.reshape(-1, self.num_token))
            o4 = F.log_softmax(o4, -1).reshape(-1, self.num_token)

            m4 = self.h(logtis_t_shuffle.reshape(-1, self.num_token))
            m4 = F.log_softmax(m4, -1).reshape(-1, self.num_token)

            pre_cls = F.log_softmax(self.f(x), -1)

            gt = gt.reshape(-1, self.num_token)
            label = gt.argmax(-1).long()  # B Gt
            label_shuffle = label.view(feature_t.shape[0], -1)[idx, :].view(-1)


            # gt_tran = F.log_softmax(logits_t, -1).reshape(-1, self.num_token)
            # gt_tran_shuffle = torch.mean(F.log_softmax(logtis_t_shuffle, -1), dim=1).reshape(-1, self.num_token)
            # return pre_cls, (gt_tran, gt_tran_shuffle, o4, m4), (o3, m3)
            return pre_cls, (o4, m4, label, label_shuffle), (o3, m3)
        else:
            pre_cls = F.log_softmax(self.f(x), -1)
            return pre_cls


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target):
        loss = F.nll_loss(pred, target)
        # mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss
        # total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
