import torch.nn as nn
import torch
import torch.nn.functional as F
from knn_cuda import KNN
#from KNN_CUDA.knn_cuda import KNN
from pointnet2_ops import pointnet2_utils
#from Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from utils import misc
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from extensions.emd import emd
from .build import MODELS
# def knn(x, k):
#     inner = -2 * torch.matmul(x.transpose(2, 1), x)
#     xx = torch.sum(x ** 2, dim=1, keepdim=True)
#     pairwise_distance = -xx - inner - xx.transpose(2, 1)
#
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
#     return idx

# from knn_cuda import KNN
knn = KNN(k=4, transpose_mode=False)

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, channel=[3, 64, 128, 1024]):
        super(PointNetEncoder, self).__init__()
        c1, c2, c3, c4 = channel
        self.conv1 = torch.nn.Conv1d(c1, c2, 1)
        self.conv2 = torch.nn.Conv1d(c2, c3, 1)
        self.conv3 = torch.nn.Conv1d(c3, c4, 1)
        self.bn1 = nn.BatchNorm1d(c2)
        self.bn2 = nn.BatchNorm1d(c3)
        self.bn3 = nn.BatchNorm1d(c4)
        self.global_feat = global_feat

    def forward(self, x):
        x = x.transpose(1,2)
        B, D, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.global_feat:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1)


class DGCNN(nn.Module):
    def __init__(self, encoder_channel, output_channel):
        super().__init__()
        '''
        K has to be 16
        '''
        self.input_trans = nn.Conv1d(encoder_channel, 128, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 256),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 512),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 512),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 1024),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer5 = nn.Sequential(nn.Conv1d(2304, output_channel, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, output_channel),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):
        # coor: bs, 3, np, x: bs, c, np

        k = 4
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = knn(coor_k, coor_q)  # bs k np  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, f, coor):
        # f: B G C
        # coor: B G 3

        # bs 3 N   bs C N
        feature_list = []
        coor = coor.transpose(1, 2).contiguous()  # B 3 N
        f = f.transpose(1, 2).contiguous()  # B C N
        f = self.input_trans(f)  # B 128 N

        f = self.get_graph_feature(coor, f, coor, f)  # B 256 N k
        f = self.layer1(f)  # B 256 N k
        f = f.max(dim=-1, keepdim=False)[0]  # B 256 N
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f)  # B 512 N k
        f = self.layer2(f)  # B 512 N k
        f = f.max(dim=-1, keepdim=False)[0]  # B 512 N
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f)  # B 1024 N k
        f = self.layer3(f)  # B 512 N k
        f = f.max(dim=-1, keepdim=False)[0]  # B 512 N
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f)  # B 1024 N k
        f = self.layer4(f)  # B 1024 N k
        f = f.max(dim=-1, keepdim=False)[0]  # B 1024 N
        feature_list.append(f)

        f = torch.cat(feature_list, dim=1)  # B 2304 N

        f = self.layer5(f)  # B C' N

        f = f.transpose(-1, -2)

        return f


### ref https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py ###
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, channel = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, channel).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Encoder(nn.Module):
    def __init__(self, input_channel, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(input_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, channel = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, channel)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Decoder(nn.Module):
    def __init__(self, encoder_channel, num_fine):
        super().__init__()
        self.num_fine = num_fine
        self.grid_size = 2
        self.num_coarse = self.num_fine // 4
        assert num_fine % 4 == 0

        self.mlp = nn.Sequential(
            nn.Linear(encoder_channel, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(encoder_channel + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(
            self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(
            self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2)  # 1 2 S

    def forward(self, feature_global):
        '''
            feature_global : B G C
            -------
            coarse : B G M 3
            fine : B G N 3

        '''
        bs, g, c = feature_global.shape
        feature_global = feature_global.reshape(bs * g, c)

        coarse = self.mlp(feature_global).reshape(bs * g, self.num_coarse, 3)  # BG M 3

        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)  # BG (M) S 3
        point_feat = point_feat.reshape(bs * g, self.num_fine, 3).transpose(2, 1)  # BG 3 N

        seed = self.folding_seed.unsqueeze(2).expand(bs * g, -1, self.num_coarse, -1)  # BG 2 M (S)
        seed = seed.reshape(bs * g, -1, self.num_fine).to(feature_global.device)  # BG 2 N

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_fine)  # BG 1024 N
        feat = torch.cat([feature_global, seed, point_feat], dim=1)  # BG C N

        center = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)  # BG (M) S 3
        center = center.reshape(bs * g, self.num_fine, 3).transpose(2, 1)  # BG 3 N

        fine = self.final_conv(feat) + center  # BG 3 N
        fine = fine.reshape(bs, g, 3, self.num_fine).transpose(-1, -2)
        coarse = coarse.reshape(bs, g, self.num_coarse, 3)
        return coarse, fine

@MODELS.register_module()
class DiscreteVAE(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.tokens_dims = config.tokens_dims

        self.decoder_dims = config.decoder_dims
        self.num_tokens = config.num_tokens

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(input_channel=3, encoder_channel=self.encoder_dims)
        self.dgcnn_1 = DGCNN(encoder_channel=self.encoder_dims, output_channel=self.num_tokens)
        self.codebook = nn.Parameter(torch.randn(self.num_tokens, self.tokens_dims))

        self.dgcnn_2 = DGCNN(encoder_channel=self.tokens_dims, output_channel=self.decoder_dims)
        self.decoder = Decoder(encoder_channel=self.decoder_dims, num_fine=self.group_size)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func_cdl1 = ChamferDistanceL1().cuda()
        self.loss_func_cdl2 = ChamferDistanceL2().cuda()
        self.loss_func_emd = emd().cuda()

    def recon_loss(self, ret, gt):
        whole_coarse, whole_fine, coarse, fine, group_gt, _ = ret

        bs, g, _, _ = coarse.shape

        coarse = coarse.reshape(bs * g, -1, 3).contiguous()
        fine = fine.reshape(bs * g, -1, 3).contiguous()
        group_gt = group_gt.reshape(bs * g, -1, 3).contiguous()

        loss_coarse_block = self.loss_func_cdl1(coarse, group_gt)
        loss_fine_block = self.loss_func_cdl1(fine, group_gt)

        loss_recon = loss_coarse_block + loss_fine_block

        return loss_recon

    def get_loss(self, ret, gt):
        # reconstruction loss
        loss_recon = self.recon_loss(ret, gt)
        # kl divergence
        logits = ret[-1]  # B G N
        softmax = F.softmax(logits, dim=-1)
        mean_softmax = softmax.mean(dim=1)
        log_qy = torch.log(mean_softmax)
        log_uniform = torch.log(torch.tensor([1. / self.num_tokens], device=gt.device))
        loss_klv = F.kl_div(log_qy, log_uniform.expand(log_qy.size(0), log_qy.size(1)), None, None, 'batchmean',
                            log_target=True)

        return loss_recon, loss_klv

    def forward(self, inp, temperature=1., hard=False, **kwargs):
        neighborhood, center = self.group_divider(inp)
        logits = self.encoder(neighborhood)  # B G C
        logits = self.dgcnn_1(logits, center)  # B G N
        soft_one_hot = F.gumbel_softmax(logits, tau=temperature, dim=2, hard=hard)  # B G N
        sampled = torch.einsum('b g n, n c -> b g c', soft_one_hot, self.codebook)  # B G C
        feature = self.dgcnn_2(sampled, center)
        coarse, fine = self.decoder(feature)

        with torch.no_grad():
            whole_fine = (fine + center.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            whole_coarse = (coarse + center.unsqueeze(2)).reshape(inp.size(0), -1, 3)

        assert fine.size(2) == self.group_size
        ret = (whole_coarse, whole_fine, coarse, fine, neighborhood, logits)
        return ret


@MODELS.register_module()
class DiscreteVAEH(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.group_size_b, self.group_size_t = config.group_size
        self.num_group_b, self.num_group_t = config.num_group
        self.encoder_dims_b, self.encoder_dims_t = config.encoder_dims
        self.tokens_dims_b, self.tokens_dims_t = config.tokens_dims
        self.num_tokens_b, self.num_tokens_t = config.num_tokens
        self.decoder_dims = config.decoder_dims

        self.group_divider_b = Group(num_group=self.num_group_b, group_size=self.group_size_b)
        self.group_divider_t = Group(num_group=self.num_group_t, group_size=self.group_size_t)

        self.encoder_b = Encoder(input_channel=3, encoder_channel=self.encoder_dims_b)
        self.dgcnn_b = DGCNN(encoder_channel=self.encoder_dims_b, output_channel=self.num_tokens_b)

        self.encoder_t = Encoder(input_channel=self.encoder_dims_b, encoder_channel=self.encoder_dims_t)
        self.dgcnn_t = DGCNN(encoder_channel=self.encoder_dims_t, output_channel=self.num_tokens_t)

        self.codebook_b = nn.Parameter(torch.randn(self.num_tokens_b, self.tokens_dims_b))
        self.codebook_t = nn.Parameter(torch.randn(self.num_tokens_t, self.tokens_dims_t))

        self.dgcnn_dec1 = DGCNN(encoder_channel=self.tokens_dims_b, output_channel=self.decoder_dims)
        self.dgcnn_dec2 = DGCNN(encoder_channel=self.tokens_dims_b+self.tokens_dims_t, output_channel=self.decoder_dims)

        self.decoder2 = Decoder(encoder_channel=self.decoder_dims, num_fine=self.group_size_b)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func_cdl1 = ChamferDistanceL1().cuda()
        self.loss_func_cdl2 = ChamferDistanceL2().cuda()
        self.loss_func_emd = emd().cuda()

    def recon_loss(self, ret):
        fine_t_pos, coarse_t_pos, group_gt = ret[2], ret[3], ret[4]

        bs, g, _, _ = fine_t_pos.shape

        coarse_t_pos = coarse_t_pos.reshape(bs * g, -1, 3).contiguous()
        fine_t_pos = fine_t_pos.reshape(bs * g, -1, 3).contiguous()
        group_gt = group_gt.reshape(bs * g, -1, 3).contiguous()

        loss_coarse_t_pos = self.loss_func_cdl1(coarse_t_pos, group_gt)
        loss_fine_t_pos = self.loss_func_cdl1(fine_t_pos, group_gt)

        loss_recon = (loss_coarse_t_pos, loss_fine_t_pos)

        return loss_recon

    def get_loss(self, ret):
        # reconstruction loss
        loss_recon = self.recon_loss(ret)
        # kl divergence
        logits_b, logits_t = ret[-2], ret[-1]

        softmax_b = F.softmax(logits_b, dim=-1)  # B Gb Nb
        softmax_t = F.softmax(logits_t, dim=-1)  # B Nt

        mean_softmax_b = softmax_b.mean(dim=1)  # B Nb

        log_qy_b = torch.log(mean_softmax_b)
        log_qy_t = torch.log(softmax_t)

        log_uniform_b = torch.log(torch.tensor([1. / self.num_tokens_b], device=torch.device('cuda')))
        log_uniform_t = torch.log(torch.tensor([1. / self.num_tokens_t], device=torch.device('cuda')))

        loss_klv_b = F.kl_div(log_qy_b, log_uniform_b.expand(log_qy_b.size(0), log_qy_b.size(1)), None, None, 'batchmean',
                            log_target=True)
        loss_klv_t = F.kl_div(log_qy_t, log_uniform_t.expand(log_qy_t.size(0), log_qy_t.size(1)), None, None, 'batchmean',
                              log_target=True)
        loss_klv = (loss_klv_b, loss_klv_t)
        return loss_recon, loss_klv

    def forward(self, inp, temperature=1., hard=False, **kwargs):
        Gb, Nb = self.num_group_b, self.num_tokens_b
        # Gt, Nt = self.num_group_t, self.num_tokens_t
        # Cb, Ct = self.encoder_dims_b, self.encoder_dims_t

        neighborhood_b, center_b = self.group_divider_b(inp)  # B Gb Gb/2 3;  B Gb 3
        feature_b = self.encoder_b(neighborhood_b)   # B Gb Cb
        logits_b = self.dgcnn_b(feature_b, center_b)  # B Gb Nb

        neighborhood_t, center_t = self.group_divider_t(feature_b.contiguous())  # B Gt Gt/2 Cb;  B Gt Cb
        feature_t = self.encoder_t(neighborhood_t)  # B Gt Ct;
        logits_t = self.dgcnn_t(feature_t, center_t)  # B Gt Nt
        logits_t = torch.mean(logits_t, dim=1)  # B Nt


        soft_one_hot_b = F.gumbel_softmax(logits_b, tau=temperature, dim=2, hard=hard) # B Gb Nb
        sampled_b = torch.einsum('b g n, n c -> b g c', soft_one_hot_b, self.codebook_b)  # B Gb Cb

        soft_one_hot_t = F.gumbel_softmax(logits_t, tau=temperature, dim=1, hard=hard)  # B Nt
        sampled_t = torch.einsum('b n, n c -> b c', soft_one_hot_t, self.codebook_t)  # B Ct

        #feature_untran = self.dgcnn_dec1(sampled_b, center_b)
        #coarse_b_pos, fine_b_neg = self.decoder1(feature_untran)

        sampled_cat = torch.cat((sampled_t.unsqueeze(1).repeat(1, Gb, 1).contiguous(), sampled_b), -1)  # B Gb Ct+Cb
        feature_tran = self.dgcnn_dec2(sampled_cat, center_b)
        coarse_t_pos, fine_t_pos = self.decoder2(feature_tran)


        with torch.no_grad():
            #whole_fine_b_neg = (fine_b_neg + center_b.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            #whole_coarse_b_pos = (coarse_b_pos + center_b.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            whole_fine_t_pos = (fine_t_pos + center_b.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            whole_coarse_t_pos = (coarse_t_pos + center_b.unsqueeze(2)).reshape(inp.size(0), -1, 3)

        ret = (whole_fine_t_pos, whole_coarse_t_pos,
               fine_t_pos, coarse_t_pos,
               neighborhood_b, logits_b, logits_t)
        return ret


@MODELS.register_module()
class DiscreteVAEHDR(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.group_size_b, self.group_size_t = config.group_size
        self.num_group_b, self.num_group_t = config.num_group
        self.encoder_dims_b, self.encoder_dims_t = config.encoder_dims
        self.tokens_dims_b, self.tokens_dims_t = config.tokens_dims
        self.num_tokens_b, self.num_tokens_t = config.num_tokens
        self.decoder_dims = config.decoder_dims


        self.group_divider_b = Group(num_group=self.num_group_b, group_size=self.group_size_b)
        self.group_divider_t = Group(num_group=self.num_group_t, group_size=self.group_size_t)

        self.encoder_b = Encoder(input_channel=3, encoder_channel=self.encoder_dims_b)
        self.dgcnn_b = DGCNN(encoder_channel=self.encoder_dims_b, output_channel=self.num_tokens_b)

        self.encoder_t = Encoder(input_channel=self.encoder_dims_b, encoder_channel=self.encoder_dims_t)
        self.dgcnn_t = DGCNN(encoder_channel=self.encoder_dims_t, output_channel=self.num_tokens_t)

        self.codebook_b = nn.Parameter(torch.randn(self.num_tokens_b, self.tokens_dims_b))
        self.codebook_t = nn.Parameter(torch.randn(self.num_tokens_t, self.tokens_dims_t))

        self.dgcnn_dec1 = DGCNN(encoder_channel=self.tokens_dims_b, output_channel=self.decoder_dims)
        self.dgcnn_dec2 = DGCNN(encoder_channel=self.tokens_dims_b+self.tokens_dims_t, output_channel=self.decoder_dims)

        self.decoder1 = Decoder(encoder_channel=self.decoder_dims, num_fine=self.group_size_b)
        self.decoder2 = Decoder(encoder_channel=self.decoder_dims, num_fine=self.group_size_b)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func_cdl1 = ChamferDistanceL1().cuda()
        self.loss_func_cdl2 = ChamferDistanceL2().cuda()
        self.loss_func_emd = emd().cuda()

    def recon_loss(self, ret):
        fine_b_neg, coarse_b_pos, fine_t_pos, coarse_t_pos, group_gt = ret[4], ret[5], ret[6], ret[7], ret[8]

        bs, g, _, _ = coarse_b_pos.shape

        coarse_b_pos = coarse_b_pos.reshape(bs * g, -1, 3).contiguous()
        coarse_t_pos = coarse_t_pos.reshape(bs * g, -1, 3).contiguous()
        fine_b_neg = fine_b_neg.reshape(bs * g, -1, 3).contiguous()
        fine_t_pos = fine_t_pos.reshape(bs * g, -1, 3).contiguous()
        group_gt = group_gt.reshape(bs * g, -1, 3).contiguous()
        loss_coarse_b_pos = self.loss_func_cdl1(coarse_b_pos, group_gt)
        # loss_fine_b_neg = self.loss_func_cdl1(fine_b_neg, group_gt)
        loss_coarse_t_pos = self.loss_func_cdl1(coarse_t_pos, group_gt)
        loss_fine_t_pos = self.loss_func_cdl1(fine_t_pos, group_gt)
        # if loss_fine_b_neg.item() > 0.5:
        #     pass
        # elif loss_fine_b_neg.item() < 0.2:
        #     loss_fine_b_neg = -loss_fine_b_neg
        # else:
        # loss_fine_b_neg = loss_fine_b_neg.detach()
        loss_fine_b_neg = torch.tensor(0.0).cuda()
        # loss_fine_b_neg = torch.max(loss_fine_b_neg-0.5, 0)[0] - torch.min(loss_fine_b_neg, 0)[0]

        loss_recon = (loss_coarse_b_pos, loss_fine_b_neg, loss_coarse_t_pos, loss_fine_t_pos)

        return loss_recon

    def get_loss(self, ret):
        # reconstruction loss
        loss_recon = self.recon_loss(ret)
        # kl divergence
        logits_b, logits_t = ret[-2], ret[-1]

        softmax_b = F.softmax(logits_b, dim=-1)  # B Gb Nb
        softmax_t = F.softmax(logits_t, dim=-1)  # B Nt

        mean_softmax_b = softmax_b.mean(dim=1)  # B Nb

        log_qy_b = torch.log(mean_softmax_b)
        log_qy_t = torch.log(softmax_t)

        log_uniform_b = torch.log(torch.tensor([1. / self.num_tokens_b], device=torch.device('cuda')))
        log_uniform_t = torch.log(torch.tensor([1. / self.num_tokens_t], device=torch.device('cuda')))

        loss_klv_b = F.kl_div(log_qy_b, log_uniform_b.expand(log_qy_b.size(0), log_qy_b.size(1)), None, None, 'batchmean',
                            log_target=True)
        loss_klv_t = F.kl_div(log_qy_t, log_uniform_t.expand(log_qy_t.size(0), log_qy_t.size(1)), None, None, 'batchmean',
                              log_target=True)
        loss_klv = (loss_klv_b, loss_klv_t)
        return loss_recon, loss_klv

    def forward(self, inp, temperature=1., hard=False, **kwargs):
        Gb, Nb = self.num_group_b, self.num_tokens_b
        # Gt, Nt = self.num_group_t, self.num_tokens_t
        # Cb, Ct = self.encoder_dims_b, self.encoder_dims_t

        neighborhood_b, center_b = self.group_divider_b(inp)  # B Gb Gb/2 3;  B Gb 3
        feature_b = self.encoder_b(neighborhood_b)   # B Gb Cb
        logits_b = self.dgcnn_b(feature_b, center_b)  # B Gb Nb

        neighborhood_t, center_t = self.group_divider_t(feature_b.contiguous())  # B Gt Gt/2 Cb;  B Gt Cb
        feature_t = self.encoder_t(neighborhood_t)  # B Gt Ct;
        logits_t = self.dgcnn_t(feature_t, center_t)  # B Gt Nt
        logits_t = torch.mean(logits_t, dim=1)  # B Nt


        soft_one_hot_b = F.gumbel_softmax(logits_b, tau=temperature, dim=2, hard=hard) # B Gb Nb
        sampled_b = torch.einsum('b g n, n c -> b g c', soft_one_hot_b, self.codebook_b)  # B Gb Cb

        soft_one_hot_t = F.gumbel_softmax(logits_t, tau=temperature, dim=1, hard=hard)  # B Nt
        sampled_t = torch.einsum('b n, n c -> b c', soft_one_hot_t, self.codebook_t)  # B Ct

        feature_untran = self.dgcnn_dec1(sampled_b, center_b)
        coarse_b_pos, fine_b_neg = self.decoder1(feature_untran)

        sampled_cat = torch.cat((sampled_t.unsqueeze(1).repeat(1, Gb, 1).contiguous(), sampled_b), -1)  # B Gb Ct+Cb
        feature_tran = self.dgcnn_dec2(sampled_cat, center_b)
        coarse_t_pos, fine_t_pos = self.decoder2(feature_tran)


        with torch.no_grad():
            whole_fine_b_neg = (fine_b_neg + center_b.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            whole_coarse_b_pos = (coarse_b_pos + center_b.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            whole_fine_t_pos = (fine_t_pos + center_b.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            whole_coarse_t_pos = (coarse_t_pos + center_b.unsqueeze(2)).reshape(inp.size(0), -1, 3)

        ret = (whole_fine_b_neg, whole_coarse_b_pos, whole_fine_t_pos, whole_coarse_t_pos,
               fine_b_neg, coarse_b_pos, fine_t_pos, coarse_t_pos,
               neighborhood_b, logits_b, logits_t)
        return ret


@MODELS.register_module()
class DiscreteVAEHm(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.group_size_b, self.group_size_t = config.group_size
        self.num_group_b, self.num_group_t = config.num_group
        self.encoder_dims_b, self.encoder_dims_t = config.encoder_dims
        self.tokens_dims_b, self.tokens_dims_t = config.tokens_dims
        self.num_tokens_b, self.num_tokens_t = config.num_tokens
        self.decoder_dims = config.decoder_dims

        self.group_divider_b = Group(num_group=self.num_group_b, group_size=self.group_size_b)
        self.group_divider_t = Group(num_group=self.num_group_t, group_size=self.group_size_t)

        self.encoder_b = Encoder(input_channel=3, encoder_channel=self.encoder_dims_b)
        self.dgcnn_b = DGCNN(encoder_channel=self.encoder_dims_b, output_channel=self.num_tokens_b)

        self.encoder_t = Encoder(input_channel=self.encoder_dims_b, encoder_channel=self.encoder_dims_t)
        self.dgcnn_t = DGCNN(encoder_channel=self.encoder_dims_t, output_channel=self.num_tokens_t)

        self.codebook_b = nn.Parameter(torch.randn(self.num_tokens_b, self.tokens_dims_b))
        self.codebook_t = nn.Parameter(torch.randn(self.num_tokens_t, self.tokens_dims_t))

        self.upsample = nn.ConvTranspose1d(in_channels=self.encoder_dims_t, out_channels=self.encoder_dims_t,
                                           kernel_size=4, stride=4)

        self.dgcnn_dec1 = DGCNN(encoder_channel=self.tokens_dims_b, output_channel=self.decoder_dims)
        self.dgcnn_dec2 = DGCNN(encoder_channel=self.tokens_dims_b + self.tokens_dims_t,
                                output_channel=self.decoder_dims)

        self.decoder2 = Decoder(encoder_channel=self.decoder_dims, num_fine=self.group_size_b)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func_cdl1 = ChamferDistanceL1().cuda()
        self.loss_func_cdl2 = ChamferDistanceL2().cuda()
        self.loss_func_emd = emd().cuda()

    def recon_loss(self, ret):
        fine_t_pos, coarse_t_pos, group_gt = ret[2], ret[3], ret[4]

        bs, g, _, _ = fine_t_pos.shape

        coarse_t_pos = coarse_t_pos.reshape(bs * g, -1, 3).contiguous()
        fine_t_pos = fine_t_pos.reshape(bs * g, -1, 3).contiguous()
        group_gt = group_gt.reshape(bs * g, -1, 3).contiguous()

        loss_coarse_t_pos = self.loss_func_cdl1(coarse_t_pos, group_gt)
        loss_fine_t_pos = self.loss_func_cdl1(fine_t_pos, group_gt)

        loss_recon = (loss_coarse_t_pos, loss_fine_t_pos)

        return loss_recon

    def get_loss(self, ret):
        # reconstruction loss
        loss_recon = self.recon_loss(ret)
        # kl divergence
        logits_b, logits_t = ret[-2], ret[-1]

        softmax_b = F.softmax(logits_b, dim=-1)  # B Gb Nb
        softmax_t = F.softmax(logits_t, dim=-1)  # B Nt

        mean_softmax_b = softmax_b.mean(dim=1)  # B Nb
        mean_softmax_t = softmax_t.mean(dim=1)  # B Nt

        log_qy_b = torch.log(mean_softmax_b)
        log_qy_t = torch.log(mean_softmax_t)

        log_uniform_b = torch.log(torch.tensor([1. / self.num_tokens_b], device=torch.device('cuda')))
        log_uniform_t = torch.log(torch.tensor([1. / self.num_tokens_t], device=torch.device('cuda')))

        loss_klv_b = F.kl_div(log_qy_b, log_uniform_b.expand(log_qy_b.size(0), log_qy_b.size(1)), None, None,
                              'batchmean',
                              log_target=True)
        loss_klv_t = F.kl_div(log_qy_t, log_uniform_t.expand(log_qy_t.size(0), log_qy_t.size(1)), None, None,
                              'batchmean',
                              log_target=True)
        loss_klv = (loss_klv_b, loss_klv_t)
        return loss_recon, loss_klv

    def forward(self, inp, temperature=1., hard=False, **kwargs):
        Gb, Nb = self.num_group_b, self.num_tokens_b
        # Gt, Nt = self.num_group_t, self.num_tokens_t
        # Cb, Ct = self.encoder_dims_b, self.encoder_dims_t

        neighborhood_b, center_b = self.group_divider_b(inp)  # B Gb Gb/2 3;  B Gb 3
        feature_b = self.encoder_b(neighborhood_b)  # B Gb Cb
        logits_b = self.dgcnn_b(feature_b, center_b)  # B Gb Nb

        neighborhood_t, center_t = self.group_divider_t(feature_b.contiguous())  # B Gt Gt/2 Cb;  B Gt Cb
        feature_t = self.encoder_t(neighborhood_t)  # B Gt Ct;
        logits_t = self.dgcnn_t(feature_t, center_t)  # B Gt Nt

        soft_one_hot_b = F.gumbel_softmax(logits_b, tau=temperature, dim=2, hard=hard)  # B Gb Nb
        sampled_b = torch.einsum('b g n, n c -> b g c', soft_one_hot_b, self.codebook_b)  # B Gb Cb

        soft_one_hot_t = F.gumbel_softmax(logits_t, tau=temperature, dim=2, hard=hard)  # B Gt Nt
        sampled_t = torch.einsum('b g n, n c -> b g c', soft_one_hot_t, self.codebook_t)  # B Gt Ct

        # feature_untran = self.dgcnn_dec1(sampled_b, center_b)
        # coarse_b_pos, fine_b_neg = self.decoder1(feature_untran)
        sampled_t_upsampled = self.upsample(sampled_t.transpose(1,2)).transpose(1,2) # B Gb=(2*Gt) Ct
        sampled_cat = torch.cat((sampled_t_upsampled, sampled_b), -1)  # B Gb Ct+Cb

        feature_tran = self.dgcnn_dec2(sampled_cat, center_b)
        coarse_t_pos, fine_t_pos = self.decoder2(feature_tran)

        with torch.no_grad():
            # whole_fine_b_neg = (fine_b_neg + center_b.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            # whole_coarse_b_pos = (coarse_b_pos + center_b.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            whole_fine_t_pos = (fine_t_pos + center_b.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            whole_coarse_t_pos = (coarse_t_pos + center_b.unsqueeze(2)).reshape(inp.size(0), -1, 3)

        ret = (whole_fine_t_pos, whole_coarse_t_pos,
               fine_t_pos, coarse_t_pos,
               neighborhood_b, logits_b, logits_t)
        return ret


@MODELS.register_module()
class DiscreteVAEHDRm(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.group_size_b, self.group_size_t = config.group_size
        self.num_group_b, self.num_group_t = config.num_group
        self.encoder_dims_b, self.encoder_dims_t = config.encoder_dims
        self.tokens_dims_b, self.tokens_dims_t = config.tokens_dims
        self.num_tokens_b, self.num_tokens_t = config.num_tokens
        self.decoder_dims = config.decoder_dims

        self.group_divider_b = Group(num_group=self.num_group_b, group_size=self.group_size_b)
        self.group_divider_t = Group(num_group=self.num_group_t, group_size=self.group_size_t)

        self.encoder_b = Encoder(input_channel=3, encoder_channel=self.encoder_dims_b)
        self.dgcnn_b = DGCNN(encoder_channel=self.encoder_dims_b, output_channel=self.num_tokens_b)

        self.encoder_t = Encoder(input_channel=self.encoder_dims_b, encoder_channel=self.encoder_dims_t)
        self.dgcnn_t = DGCNN(encoder_channel=self.encoder_dims_t, output_channel=self.num_tokens_t)

        self.codebook_b = nn.Parameter(torch.randn(self.num_tokens_b, self.tokens_dims_b))
        self.codebook_t = nn.Parameter(torch.randn(self.num_tokens_t, self.tokens_dims_t))

        #self.upsample = nn.ConvTranspose1d(in_channels=self.encoder_dims_t, out_channels=self.encoder_dims_t, kernel_size=4, stride=4)

        self.dgcnn_dec1 = DGCNN(encoder_channel=self.tokens_dims_b, output_channel=self.decoder_dims)
        self.dgcnn_dec2 = DGCNN(encoder_channel=self.tokens_dims_b + self.tokens_dims_t,
                                output_channel=self.decoder_dims)

        self.decoder1 = Decoder(encoder_channel=self.decoder_dims, num_fine=self.group_size_b)
        self.decoder2 = Decoder(encoder_channel=self.decoder_dims, num_fine=self.group_size_b)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func_cdl1 = ChamferDistanceL1().cuda()
        self.loss_func_cdl2 = ChamferDistanceL2().cuda()
        self.loss_func_emd = emd().cuda()

    def recon_loss(self, ret):
        fine_b_neg, coarse_b_pos, fine_t_pos, coarse_t_pos, group_gt = ret[4], ret[5], ret[6], ret[7], ret[8]

        bs, g, _, _ = coarse_b_pos.shape

        coarse_b_pos = coarse_b_pos.reshape(bs * g, -1, 3).contiguous()
        coarse_t_pos = coarse_t_pos.reshape(bs * g, -1, 3).contiguous()
        fine_b_neg = fine_b_neg.reshape(bs * g, -1, 3).contiguous()
        fine_t_pos = fine_t_pos.reshape(bs * g, -1, 3).contiguous()
        group_gt = group_gt.reshape(bs * g, -1, 3).contiguous()
        group_gt_down = group_gt[:,0:8,:].contiguous()
        loss_coarse_b_pos = self.loss_func_cdl1(coarse_b_pos, group_gt_down)
        loss_fine_b_neg = self.loss_func_cdl1(fine_b_neg, group_gt_down)
        loss_coarse_t_pos = self.loss_func_cdl1(coarse_t_pos, group_gt)
        loss_fine_t_pos = self.loss_func_cdl1(fine_t_pos, group_gt)
        
        # if loss_fine_b_neg.item() > 0.5:
        #     pass
        # elif loss_fine_b_neg.item() < 0.2:
        #     loss_fine_b_neg = -loss_fine_b_neg
        # else:
        # loss_fine_b_neg = loss_fine_b_neg.detach()
        #loss_fine_b_neg = torch.tensor(0.0).cuda()
        # loss_fine_b_neg = torch.max(loss_fine_b_neg-0.5, 0)[0] - torch.min(loss_fine_b_neg, 0)[0]

        loss_recon = (loss_coarse_b_pos, loss_fine_b_neg, loss_coarse_t_pos, loss_fine_t_pos)

        return loss_recon

    def get_loss(self, ret):
        # reconstruction loss
        loss_recon = self.recon_loss(ret)
        # kl divergence
        logits_b, logits_t = ret[-2], ret[-1]

        softmax_b = F.softmax(logits_b, dim=-1)  # B G Nb
        softmax_t = F.softmax(logits_t, dim=-1)  # B G Nt

        mean_softmax_b = softmax_b.mean(dim=1)  # B Nb
        mean_softmax_t = softmax_t.mean(dim=1)  # B Nt

        log_qy_b = torch.log(mean_softmax_b)
        log_qy_t = torch.log(mean_softmax_t)

        log_uniform_b = torch.log(torch.tensor([1. / self.num_tokens_b], device=torch.device('cuda')))
        log_uniform_t = torch.log(torch.tensor([1. / self.num_tokens_t], device=torch.device('cuda')))

        loss_klv_b = F.kl_div(log_qy_b, log_uniform_b.expand(log_qy_b.size(0), log_qy_b.size(1)), None, None,
                              'batchmean',
                              log_target=True)
        loss_klv_t = F.kl_div(log_qy_t, log_uniform_t.expand(log_qy_t.size(0), log_qy_t.size(1)), None, None,
                              'batchmean',
                              log_target=True)
        loss_klv = (loss_klv_b, loss_klv_t)
        return loss_recon, loss_klv

    def forward(self, inp, temperature=1., hard=False, **kwargs):
        Gb, Nb = self.num_group_b, self.num_tokens_b
        # Gt, Nt = self.num_group_t, self.num_tokens_t
        # Cb, Ct = self.encoder_dims_b, self.encoder_dims_t

        neighborhood_b, center_b = self.group_divider_b(inp)  # B Gb Gb/2 3;  B Gb 3
        feature_b = self.encoder_b(neighborhood_b)  # B Gb Cb
        logits_b = self.dgcnn_b(feature_b, center_b)  # B Gb Nb

        neighborhood_t, center_t = self.group_divider_t(feature_b.contiguous())  # B Gt Gt/2 Cb;  B Gt Cb
        feature_t = self.encoder_t(neighborhood_t)  # B Gt Ct;
        logits_t = self.dgcnn_t(feature_t, center_t)  # B Gt Nt

        soft_one_hot_b = F.gumbel_softmax(logits_b, tau=temperature, dim=2, hard=hard)  # B G Nb
        sampled_b = torch.einsum('b g n, n c -> b g c', soft_one_hot_b, self.codebook_b)  # B G Cb

        soft_one_hot_t = F.gumbel_softmax(logits_t, tau=temperature, dim=2, hard=hard)  # B G Nt
        sampled_t = torch.einsum('b g n, n c -> b g c', soft_one_hot_t, self.codebook_t)  # B G Ct

        feature_untran = self.dgcnn_dec1(sampled_b, center_b)
        coarse_b_pos, fine_b_neg = self.decoder1(feature_untran)

        #sampled_t_upsampled = self.upsample(sampled_t.transpose(1,2)).transpose(1,2)
        sampled_cat = torch.cat((sampled_t, sampled_b), -1)  # B G Ct+Cb

        feature_tran = self.dgcnn_dec2(sampled_cat, center_b)
        coarse_t_pos, fine_t_pos = self.decoder2(feature_tran)

        with torch.no_grad():
            whole_fine_b_neg = (fine_b_neg + center_b.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            whole_coarse_b_pos = (coarse_b_pos + center_b.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            whole_fine_t_pos = (fine_t_pos + center_b.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            whole_coarse_t_pos = (coarse_t_pos + center_b.unsqueeze(2)).reshape(inp.size(0), -1, 3)

        ret = (whole_fine_b_neg, whole_coarse_b_pos, whole_fine_t_pos, whole_coarse_t_pos,
               fine_b_neg, coarse_b_pos, fine_t_pos, coarse_t_pos,
               neighborhood_b, logits_b, logits_t)
        return ret


@MODELS.register_module()
class DiscreteVAEHDRng(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.group_size_b, self.group_size_t = config.group_size
        self.num_group_b, self.num_group_t = config.num_group
        self.encoder_dims_b, self.encoder_dims_t = config.encoder_dims
        self.tokens_dims_b, self.tokens_dims_t = config.tokens_dims
        self.num_tokens_b, self.num_tokens_t = config.num_tokens
        self.decoder_dims = config.decoder_dims

        self.encoder_b = PointNetEncoder(global_feat=True, channel=[3, 64, 512, self.num_tokens_b])
        self.encoder_t = PointNetEncoder(global_feat=True, channel=[self.num_tokens_b, 512, 256, self.num_tokens_t])

        self.codebook_b = nn.Parameter(torch.randn(self.num_tokens_b, self.tokens_dims_b))
        self.codebook_t = nn.Parameter(torch.randn(self.num_tokens_t, self.tokens_dims_t))

        self.upsample = nn.ConvTranspose1d(in_channels=self.encoder_dims_t, out_channels=self.encoder_dims_t, kernel_size=4, stride=4)

        #self.dgcnn_dec1 = DGCNN(encoder_channel=self.tokens_dims_b, output_channel=self.decoder_dims)
        #self.dgcnn_dec2 = DGCNN(encoder_channel=self.tokens_dims_b + self.tokens_dims_t,
        #                        output_channel=self.decoder_dims)
                                
                      
        #self.pn_dec1 = PointNetEncoder(global_feat=True, channel=[self.tokens_dims_b,128,256,self.decoder_dims])
        #self.pn_dec2 = PointNetEncoder(global_feat=True, channel=[self.tokens_dims_b+self.tokens_dims_t,128,256,self.decoder_dims])                                          
        self.decoder1 = Decoder(encoder_channel=self.tokens_dims_b, num_fine=self.group_size_b)
        self.decoder2 = Decoder(encoder_channel=self.tokens_dims_b+self.tokens_dims_t, num_fine=self.group_size_b)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func_cdl1 = ChamferDistanceL1().cuda()
        self.loss_func_cdl2 = ChamferDistanceL2().cuda()
        self.loss_func_emd = emd().cuda()

    def recon_loss(self, ret):
        fine_b_neg, coarse_b_pos, fine_t_pos, coarse_t_pos, group_gt = ret[0], ret[1], ret[2], ret[3], ret[4]

        bs, g, _ = coarse_b_pos.shape

        coarse_b_pos = coarse_b_pos.reshape(bs * g, -1, 3).contiguous()
        coarse_t_pos = coarse_t_pos.reshape(bs * g, -1, 3).contiguous()
        fine_b_neg = fine_b_neg.reshape(bs * g, -1, 3).contiguous()
        fine_t_pos = fine_t_pos.reshape(bs * g, -1, 3).contiguous()
        group_gt = group_gt.reshape(bs * g, -1, 3).contiguous()
        loss_coarse_b_pos = self.loss_func_cdl1(coarse_b_pos, group_gt)
        # loss_fine_b_neg = self.loss_func_cdl1(fine_b_neg, group_gt)
        loss_coarse_t_pos = self.loss_func_cdl1(coarse_t_pos, group_gt)
        loss_fine_t_pos = self.loss_func_cdl1(fine_t_pos, group_gt)
        # if loss_fine_b_neg.item() > 0.5:
        #     pass
        # elif loss_fine_b_neg.item() < 0.2:
        #     loss_fine_b_neg = -loss_fine_b_neg
        # else:
        # loss_fine_b_neg = loss_fine_b_neg.detach()
        loss_fine_b_neg = torch.tensor(0.0).cuda()
        # loss_fine_b_neg = torch.max(loss_fine_b_neg-0.5, 0)[0] - torch.min(loss_fine_b_neg, 0)[0]

        loss_recon = (loss_coarse_b_pos, loss_fine_b_neg, loss_coarse_t_pos, loss_fine_t_pos)

        return loss_recon

    def get_loss(self, ret):
        # reconstruction loss
        loss_recon = self.recon_loss(ret)
        # kl divergence
        logits_b, logits_t = ret[-2], ret[-1]

        softmax_b = F.softmax(logits_b, dim=-1)  # B Gb Nb
        softmax_t = F.softmax(logits_t, dim=-1)  # B Gt Nt

        mean_softmax_b = softmax_b.mean(dim=1)  # B Nb
        mean_softmax_t = softmax_t.mean(dim=1)  # B Nt

        log_qy_b = torch.log(mean_softmax_b)
        log_qy_t = torch.log(mean_softmax_t)

        log_uniform_b = torch.log(torch.tensor([1. / self.num_tokens_b], device=torch.device('cuda')))
        log_uniform_t = torch.log(torch.tensor([1. / self.num_tokens_t], device=torch.device('cuda')))

        loss_klv_b = F.kl_div(log_qy_b, log_uniform_b.expand(log_qy_b.size(0), log_qy_b.size(1)), None, None,
                              'batchmean',
                              log_target=True)
        loss_klv_t = F.kl_div(log_qy_t, log_uniform_t.expand(log_qy_t.size(0), log_qy_t.size(1)), None, None,
                              'batchmean',
                              log_target=True)
        loss_klv = (loss_klv_b, loss_klv_t)
        return loss_recon, loss_klv

    def forward(self, inp, temperature=1., hard=False, **kwargs):
        inp = inp.float()
        B = inp.shape[0]
        Gb, Nb = self.num_group_b, self.num_tokens_b
        Gt, Nt = self.num_group_t, self.num_tokens_t
        # Cb, Ct = self.encoder_dims_b, self.encoder_dims_t

        feature_b = self.encoder_b(inp.float())  # B Nb N 
        x1 = F.adaptive_max_pool1d(feature_b, int(Gb/2)).transpose(1,2).view(B, int(Gb/2), -1)  # B Gb/2 Nb
        x2 = F.adaptive_avg_pool1d(feature_b, int(Gb/2)).transpose(1,2).view(B, int(Gb/2), -1)  # B Gb/2 Nb
        logits_b = torch.cat((x1, x2), 1)  # B Gb Nb

        feature_t = self.encoder_t(feature_b)  # B N Nt
        x1 = F.adaptive_max_pool1d(feature_t, int(Gt/2)).transpose(1,2).view(B, int(Gt/2), -1)  # B Gt/2 Nt
        x2 = F.adaptive_avg_pool1d(feature_t, int(Gt/2)).transpose(1,2).view(B, int(Gt/2), -1)  # B Gt/2 Nt
        logits_t = torch.cat((x1, x2), 1)  # B Gt Nt

        soft_one_hot_b = F.gumbel_softmax(logits_b, tau=temperature, dim=2, hard=hard)  # B Gb Nb
        sampled_b = torch.einsum('b g n, n c -> b g c', soft_one_hot_b, self.codebook_b)  # B Gb Cb

        soft_one_hot_t = F.gumbel_softmax(logits_t, tau=temperature, dim=2, hard=hard)  # B Gt Nt
        sampled_t = torch.einsum('b g n, n c -> b g c', soft_one_hot_t, self.codebook_t)  # B Gt Ct

        #feature_untran = self.pn_dec1(sampled_b).transpose(1,2)  # B Gt Ct
        coarse_b_pos, fine_b_neg = self.decoder1(sampled_b)

        sampled_t_upsampled = self.upsample(sampled_t.transpose(1, 2)).transpose(1, 2) # B Gb=(2*Gt) Ct
        sampled_cat = torch.cat((sampled_t_upsampled, sampled_b), -1)  # B Gb Ct+Cb

        #feature_tran = self.pn_dec2(sampled_cat).transpose(1,2)  # B Gb Cb
        coarse_t_pos, fine_t_pos = self.decoder2(sampled_cat)

        with torch.no_grad():
            whole_fine_b_neg = fine_b_neg.reshape(inp.size(0), -1, 3)
            whole_coarse_b_pos = coarse_b_pos.reshape(inp.size(0), -1, 3)
            whole_fine_t_pos = fine_t_pos.reshape(inp.size(0), -1, 3)
            whole_coarse_t_pos = coarse_t_pos.reshape(inp.size(0), -1, 3)

        ret = (whole_fine_b_neg, whole_coarse_b_pos, whole_fine_t_pos, whole_coarse_t_pos,
               inp,
               logits_b, logits_t)
        return ret

