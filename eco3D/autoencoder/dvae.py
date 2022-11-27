import torch.nn as nn
import torch
import torch.nn.functional as F
#from knn_cuda import KNN
from knn_cuda import KNN
#from pointnet2_ops import pointnet2_utils
from pointnet2_ops import pointnet2_utils
from utils import misc
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from extensions.emd import emd

# def knn(x, k):
#     inner = -2 * torch.matmul(x.transpose(2, 1), x)
#     xx = torch.sum(x ** 2, dim=1, keepdim=True)
#     pairwise_distance = -xx - inner - xx.transpose(2, 1)
#
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
#     return idx

# from knn_cuda import KNN
knn = KNN(k=4, transpose_mode=False)


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
        center, idx_center = misc.fps(xyz, self.num_group)  # B G 3
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
        return neighborhood, center, idx_center, idx


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


class DiscreteVAEHDRm(nn.Module):
    def __init__(self, num_token_t, group_size=[32, 8], num_group=[64, 64], encoder_dims=[256, 64], num_tokens=8192,
                 tokens_dims=[128, 16], decoder_dims=128, **kwargs):

        super().__init__()
        self.group_size_b, self.group_size_t = group_size
        self.num_group_b, self.num_group_t = num_group
        self.encoder_dims_b, self.encoder_dims_t = encoder_dims
        self.tokens_dims_b, self.tokens_dims_t = tokens_dims
        self.num_tokens_b, self.num_tokens_t = num_tokens, num_token_t
        self.decoder_dims = decoder_dims

        self.group_divider_b = Group(num_group=self.num_group_b, group_size=self.group_size_b)
        self.group_divider_t = Group(num_group=self.num_group_t, group_size=self.group_size_t)

        self.encoder_b = Encoder(input_channel=3, encoder_channel=self.encoder_dims_b)
        self.dgcnn_b = DGCNN(encoder_channel=self.encoder_dims_b, output_channel=self.num_tokens_b)

        self.encoder_t = Encoder(input_channel=self.encoder_dims_b, encoder_channel=self.encoder_dims_t)
        self.dgcnn_t = DGCNN(encoder_channel=self.encoder_dims_t, output_channel=self.num_tokens_t)

        self.codebook_b = nn.Parameter(torch.randn(self.num_tokens_b, self.tokens_dims_b))
        self.codebook_t = nn.Parameter(torch.randn(self.num_tokens_t, self.tokens_dims_t))


        self.dgcnn_dec1 = DGCNN(encoder_channel=self.tokens_dims_b, output_channel=self.decoder_dims)
        self.dgcnn_dec2 = DGCNN(encoder_channel=self.tokens_dims_b + self.tokens_dims_t,
                                output_channel=self.decoder_dims)

        self.decoder1 = Decoder(encoder_channel=self.decoder_dims, num_fine=self.group_size_b)
        self.decoder2 = Decoder(encoder_channel=self.decoder_dims, num_fine=self.group_size_b)


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

        soft_one_hot_t = F.gumbel_softmax(logits_t, tau=temperature, dim=1, hard=hard)  # B Gt Nt
        sampled_t = torch.einsum('b g n, n c -> b g c', soft_one_hot_t, self.codebook_t)  # B Gt Ct

        feature_untran = self.dgcnn_dec1(sampled_b, center_b)
        coarse_b_pos, fine_b_neg = self.decoder1(feature_untran)

        sampled_t_upsampled = self.upsample(sampled_t.transpose(1,2)).transpose(1,2) # B Gb=(2*Gt) Ct
        sampled_cat = torch.cat((sampled_t_upsampled, sampled_b), -1)  # B Gb Ct+Cb

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



