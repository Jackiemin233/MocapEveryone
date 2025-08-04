import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops import rearrange
import random
from .SWT import WaveletEmbedding


class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def add_position_emb(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return input_feat + position_embeddings


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class GMMBlock(nn.Module):
    def __init__(self, config):
        super(GMMBlock, self).__init__()
        self.attn0 = BertAttention(config)
        self.attn1 = BertAttention(config, wid=0.5)
        self.attn2 = BertAttention(config, wid=1.0)
        self.attn3 = BertAttention(config, wid=5.0)

    def forward(self, input_tensor, attention_mask=None):
        o0 = self.attn0(input_tensor, attention_mask).unsqueeze(-1)
        o1 = self.attn1(input_tensor, attention_mask).unsqueeze(-1)
        o2 = self.attn2(input_tensor, attention_mask).unsqueeze(-1)
        o3 = self.attn3(input_tensor, attention_mask).unsqueeze(-1)

        oo = torch.cat([o0, o1, o2, o3], dim=-1)
        out = torch.mean(oo, dim=-1).squeeze()

        return out


class BertAttention(nn.Module):
    def __init__(self, config, wid=None):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config, wid=wid)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, L)
        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertAttentionFusion(nn.Module):
    def __init__(self, config, wid=None):
        super(BertAttentionFusion, self).__init__()
        self.self = BertSelfAttention(config, wid=wid)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor_1, input_tensor_2, attention_mask=None):
        """
        Args:
            input_tensor_1: (N, L_1, D)
            input_tensor_2: (N, L_2, D)
            attention_mask: (N, L)

            output: (N, L_1, D)
        """
        self_output = self.self(input_tensor_1, input_tensor_2, input_tensor_2, attention_mask)
        attention_output = self.output(self_output, input_tensor_1)
        return attention_output


class BertSelfAttention(nn.Module):
    def __init__(self, config, wid=None):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (
                config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.wid = wid

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def generate_gauss_weight(self, props_len, width):

        center = torch.arange(props_len).cuda() / props_len
        width = width * torch.ones(props_len).cuda()
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / 9

        w = 0.3989422804014327

        weight = w / width * torch.exp(-(weight - center) ** 2 / (2 * width ** 2))

        return weight / weight.max(dim=-1, keepdim=True)[0]

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)
        """

        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)
        # transpose
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores_ori = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)

        attention_scores_ori = attention_scores_ori / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores_ori
        if self.wid is not None:
            gmm_mask = self.generate_gauss_weight(attention_scores.shape[-1], self.wid)
            gmm_mask = gmm_mask.unsqueeze(0).unsqueeze(0)
            attention_scores = attention_scores_ori * gmm_mask
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
            attention_scores = attention_scores + attention_mask
        # attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # compute output context
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*4),
            nn.GELU(),
            nn.Linear(config.hidden_size*4, config.hidden_size),
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)


class CQAttention(nn.Module):
    def __init__(self, dim, drop_rate=0.0):
        super(CQAttention, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query):
        score = self.trilinear_attention(context, query)  # (batch_size, c_seq_len, q_seq_len)
        score_ = nn.Softmax(dim=2)(score)  # (batch_size, c_seq_len, q_seq_len)
        score_t = nn.Softmax(dim=1)(score)  # (batch_size, c_seq_len, q_seq_len)
        score_t = score_t.transpose(1, 2)  # (batch_size, q_seq_len, c_seq_len)
        c2q = torch.matmul(score_, query)  # (batch_size, c_seq_len, dim)
        q2c = torch.matmul(torch.matmul(score_, score_t), context)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, c2q, torch.mul(context, c2q), torch.mul(context, q2c)], dim=2)
        output = self.cqa_linear(output)  # (batch_size, c_seq_len, dim)
        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand([-1, -1, q_seq_len])  # (batch_size, c_seq_len, q_seq_len)
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand([-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)
        return res


class MotionDecoder(nn.Module):
    def __init__(self, input_dim, output_joints=21, output_xyz=3):
        super(MotionDecoder, self).__init__()
        self.output_joints = output_joints
        self.output_xyz = output_xyz
        total_output_dim = output_joints * output_xyz

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, total_output_dim)

        self.mask_token = nn.Parameter(torch.randn(1, 1, input_dim))

        self.loss_fn = nn.SmoothL1Loss()

        self.map_text = nn.Linear(256, input_dim)
        self.fc = nn.Linear(input_dim*2, input_dim)
        self.LayerNorm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x形状: [batch_size, time, dim]
        # text_emb: [batch_size, dim]
        # batch_size, time_steps, dim = x.shape

        # 展平时间维度
        # x = x.reshape(batch_size * time_steps, dim)

        # 通过MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        '''x = F.relu(self.fc1(x))
        x = self.fc2(x)'''

        # 恢复原始形状 [batch_size, time, joint, xyz]
        # x = x.reshape(batch_size, time_steps, self.output_joints, self.output_xyz)
        return x

    def mask_features(self, motion_embeds, text_emb, motion_mask):
        # x形状: [batch_size, time, dim]
        batch_size, time_steps, dim = motion_embeds.shape

        text_emb = self.map_text(text_emb).unsqueeze(1).expand(-1, time_steps, -1)

        # 随机生成mask位置 (每个样本不同)
        motion_len = motion_mask.sum(dim=-1)
        mask_positions = []
        for i in range(motion_len.shape[0]):
            mask_positions.append(random.randint(0, int(motion_len[i])))
        # mask_positions = torch.randint(0, time_steps, (batch_size,))

        # 创建mask
        mask = torch.ones_like(motion_embeds)
        for i, pos in enumerate(mask_positions):
            if pos < time_steps:
                mask[i, pos:, :] = 0

        # 应用mask
        masked_embeds = motion_embeds * mask
        masked_embeds = masked_embeds + (1 - mask) * self.mask_token

        masked_embeds = self.LayerNorm(self.fc(torch.cat((masked_embeds, text_emb), dim=-1)))
        masked_embeds = F.relu(masked_embeds)

        return masked_embeds, mask

    def loss(self, reconstructed, motion_data, mask, motion_mask):
        # 计算mask部分的损失
        # 将mask从embeds形状转换为motion形状 [batch, time] -> [batch, time, joint, xyz]

        '''reconstructed = F.normalize(reconstructed, dim=-1)
        motion_data = F.normalize(motion_data, dim=-1)'''

        '''motion_mask = mask[:, :, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.output_joints, self.output_xyz)
        loss = self.loss_fn(reconstructed * (1 - motion_mask), motion_data * (1 - motion_mask))'''
        motion_data = motion_data.reshape(motion_data.shape[0], motion_data.shape[1], -1)
        motion_mask = motion_mask.unsqueeze(-1).expand(-1, -1, mask.shape[2]) - mask
        motion_mask = motion_mask[:, :, 0].unsqueeze(-1).expand(-1, -1, self.output_joints * self.output_xyz)
        loss = self.loss_fn(reconstructed * motion_mask, motion_data * motion_mask)
        # loss = self.loss_fn(reconstructed, motion_data)
        '''if random.randint(0, 1) > 0.9:
            print(reconstructed[0, 100, 10])
            print(motion_data[0, 100, 10])
            print(motion_mask[0, 100, 10])'''

        return loss


class MotionEncoder_New(nn.Module):
    def __init__(
        self,
        dim,
        n_heads=4,
        dropout=0.0,
    ) -> None:
        super().__init__()

        self.temporal_embedding = LinearLayer(22*3*2, dim, dropout=dropout)
        self.temporal_pos_embed = TrainablePositionalEncoding(max_position_embeddings=225, hidden_size=dim, dropout=dropout)

        self.spatial_embedding = LinearLayer(224 * 3, dim, dropout=dropout)
        self.spatial_pos_embed = TrainablePositionalEncoding(max_position_embeddings=21, hidden_size=dim,
                                                              dropout=dropout)
        self.spatial_encoder = BertAttention(edict(hidden_size=dim, intermediate_size=dim, hidden_dropout_prob=dropout,
                                                    num_attention_heads=n_heads, attention_probs_dropout_prob=dropout))
        self.spatial_encoder_2 = BertAttention(
            edict(hidden_size=dim, intermediate_size=dim, hidden_dropout_prob=dropout,
                  num_attention_heads=n_heads, attention_probs_dropout_prob=dropout))

        self.modular_vector_mapping = nn.Linear(dim, out_features=1, bias=False)

        self.frame_order_prediction_layers = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, 223),
        )

        # self.motion_decoder = MotionDecoder(dim)
        self.swt = WaveletEmbedding(d_channel=224, swt=True)
        self.iswt = WaveletEmbedding(d_channel=224, swt=False)
        self.temporal_embedding_reverse = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 22 * 3 * 2),
        )
        self.swt_map = nn.Sequential(
            nn.Linear(dim*4, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, dim),
        )
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, motion, training=False, motion_mask=None):  # motion: [batch_size, time, joint, xyz]
        diff = motion[:, 1:, :, :] - motion[:, :-1, :, :]  # 计算motion的差分
        diff = torch.cat((diff, torch.zeros((diff.shape[0], 1, diff.shape[2], diff.shape[3]), device=diff.device)), dim=1)
        diff = torch.cat((motion, diff), dim=-1)  # 将原序列与差分concat在一起， batch_size, time, joint, xyz * 2

        diff = diff.reshape(diff.shape[0], diff.shape[1], -1)  # 展开最后一个维度， batch_size, time, joint * xyz * 2

        diff_raw = diff  # 用来提取特征的原始输入，batch_size, time, joint * xyz * 2
        diff = self.swt(diff)  # 进行小波变换，batch_size, time, 4, joint * xyz * 2

        diff = self.temporal_embedding(diff)  # 通过MLP得到embedding，batch_size, time, 4, dim

        if not training:
            diff = self.swt_encode(diff, motion_mask)  # 进行intra、inter变化后的特征，batch_size, time, dim
            temporal_motion = self.get_modularized_temporal_motion(diff, motion_mask)  # 进行pool操作，batch_size, dim
            return temporal_motion
        else:
            swt_emb_0, swt_emb_1, swt_emb_2, swt_emb_3, ori_diff, _ = self.swt_encode(diff, motion_mask, return_all=True)  # 重建的4条基带与进行intra、inter变化后的特征，大小均为batch_size, time, dim
            reconstruct_loss = self.swt_reconstruct_intra(swt_emb_0, swt_emb_1, swt_emb_2, swt_emb_3, diff_raw) * 5

            shuffle_loss, shuffled_diff = self.clip_order_prediction(diff_raw, ori_diff, motion_mask)  # 将原始输入打乱，通过intra-inter提取特征、重建并预测原始位置
            temporal_motion = self.get_modularized_temporal_motion(ori_diff, motion_mask)  # 进行pool操作，batch_size, dim

            return temporal_motion, shuffle_loss + reconstruct_loss

    def pos_and_encode(self, features, encoder, motion_mask=None):
        if motion_mask is not None:
            motion_mask = motion_mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        features = encoder(features, motion_mask)
        return features

    def swt_encode(self, swt_emb, motion_mask, return_all=False):
        swt_emb_0, swt_emb_1, swt_emb_2, swt_emb_3 = self.intra_swt_encode(swt_emb, motion_mask) # 每条系带过Encoder
        swt_emb, ori_swt_emb = self.inter_swt_encode(swt_emb_0, swt_emb_1, swt_emb_2, swt_emb_3, motion_mask)
        # 每一个基带
        if return_all:
            return swt_emb_0, swt_emb_1, swt_emb_2, swt_emb_3, swt_emb, ori_swt_emb
        else:
            return swt_emb

    def intra_swt_encode(self, swt_emb, motion_mask):
        swt_emb_0 = self.pos_and_encode(swt_emb[:, :, 0, :], self.spatial_encoder, motion_mask)
        swt_emb_1 = self.pos_and_encode(swt_emb[:, :, 1, :], self.spatial_encoder, motion_mask)
        swt_emb_2 = self.pos_and_encode(swt_emb[:, :, 2, :], self.spatial_encoder, motion_mask)
        swt_emb_3 = self.pos_and_encode(swt_emb[:, :, 3, :], self.spatial_encoder, motion_mask)

        return swt_emb_0, swt_emb_1, swt_emb_2, swt_emb_3

    def inter_swt_encode(self, swt_emb_0, swt_emb_1, swt_emb_2, swt_emb_3, motion_mask):
        swt_emb = torch.cat((swt_emb_0, swt_emb_1, swt_emb_2, swt_emb_3), dim=-1)
        swt_emb = self.swt_map(swt_emb)
        # swt_emb = swt_emb_0 + swt_emb_1 + swt_emb_2 + swt_emb_3
        ori_swt_emb = swt_emb

        if motion_mask is not None:
            motion_mask = motion_mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor

        swt_emb = self.spatial_encoder_2(swt_emb, motion_mask)

        return swt_emb, ori_swt_emb

    def swt_reconstruct_intra(self, swt_emb_0, swt_emb_1, swt_emb_2, swt_emb_3, diff_raw):
        swt_emb_0 = self.temporal_embedding_reverse(swt_emb_0).unsqueeze(2)
        swt_emb_1 = self.temporal_embedding_reverse(swt_emb_1).unsqueeze(2)
        swt_emb_2 = self.temporal_embedding_reverse(swt_emb_2).unsqueeze(2)
        swt_emb_3 = self.temporal_embedding_reverse(swt_emb_3).unsqueeze(2)

        swt_emb = torch.cat((swt_emb_0, swt_emb_1, swt_emb_2, swt_emb_3), dim=2)#.permute(0, 3, 2, 1)
        swt_emb = self.iswt(swt_emb)#.permute(0, 2, 1)

        loss = self.loss_fn(swt_emb, diff_raw)

        return loss

    def swt_reconstruct_inter(self, swt_emb, diff_raw):  # 128 224 784
        swt_emb = self.temporal_embedding_reverse_inter(swt_emb).reshape(swt_emb.shape[0], swt_emb.shape[1], 4, -1)
        swt_emb = self.iswt(swt_emb)#.permute(0, 2, 1)
        loss = self.loss_fn(swt_emb, diff_raw)
        return loss


    def get_modularized_temporal_motion(self, temporal_motion, motion_mask):
        """
        Args:
            encoded_query: (N, L, D)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(temporal_motion)  # (N, L, 2 or 1)
        # modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, motion_mask.unsqueeze(2)), dim=1)
        modular_attention_scores = F.softmax(modular_attention_scores, dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, temporal_motion)  # (N, 2 or 1, D)
        return modular_queries.squeeze()

    def partial_shuffle(self, video, ratio=0.25):
        batch_size, clip_num, dim = video.shape
        device = video.device
        k = int(clip_num * ratio)

        # 初始化原始位置索引
        original_idx = torch.arange(clip_num, device=device).expand(batch_size, clip_num).clone()  # [B, N]

        # 生成随机掩码选择要打乱的位置
        rand_mask = torch.rand(batch_size, clip_num, device=device)
        _, selected_pos = torch.topk(rand_mask, k, dim=1)  # [B, k]

        # 为选中位置生成随机排列
        shuffle_order = torch.rand(batch_size, k, device=device).argsort(dim=1)  # [B, k]
        shuffled_pos = torch.gather(selected_pos, 1, shuffle_order)  # [B, k]

        # 更新索引矩阵
        batch_idx = torch.arange(batch_size, device=device)[:, None]  # [B, 1]
        original_idx[batch_idx, selected_pos] = shuffled_pos

        # 执行打乱
        shuffled_video = video[batch_idx, original_idx, :]
        return shuffled_video, original_idx,

    def generate_tensor(self, bs, clip_num, n):  # n为分组数量，可参考MM那篇的写法
        # 创建基础模式 [0,0..., 1,1..., ..., n-1,n-1...]
        pattern = torch.arange(n).repeat_interleave(int(clip_num/n))  # 形状: (clip_num,)
        # 扩展至目标维度
        return pattern.unsqueeze(0).expand(bs, -1)  # 形状: [bs, clip_num]

    def clip_order_prediction(self, clip_video_feat, original_encoded_vid_proposal_feat, motion_mask):  # 输入：原始输入序列、原始特征、mask
        shuffle_clip_video_feat, shuffle_idx = self.partial_shuffle(clip_video_feat)  #  打乱的输入序列与打乱的坐标

        original_idx = self.generate_tensor(clip_video_feat.shape[0], clip_video_feat.shape[1], n=16).to(clip_video_feat.device)  # 原始坐标
        batch_idx = torch.arange(clip_video_feat.shape[0], device=clip_video_feat.device)[:, None]  # [batch_size, 1]
        shuffle_idx = original_idx[batch_idx, shuffle_idx]  # 打乱的坐标
        all_clip_sum = clip_video_feat.shape[0] * clip_video_feat.shape[1]

        # 乱序输入的提取特征、重建
        diff_raw = shuffle_clip_video_feat
        shuffle_clip_video_feat = self.swt(shuffle_clip_video_feat)
        shuffle_clip_video_feat = self.temporal_embedding(shuffle_clip_video_feat)
        swt_emb_0, swt_emb_1, swt_emb_2, swt_emb_3, shuffle_clip_video_feat, _ = self.swt_encode(shuffle_clip_video_feat, motion_mask, return_all=True)
        reconstruct_loss = self.swt_reconstruct_intra(swt_emb_0, swt_emb_1, swt_emb_2, swt_emb_3, diff_raw) * 5

        # 预测乱序特征位置
        shuffle_predicted_order = self.frame_order_prediction_layers(shuffle_clip_video_feat)  # [bs, 32, 32]
        shuffle_prediction_loss = F.cross_entropy(shuffle_predicted_order.view(all_clip_sum, -1), shuffle_idx.reshape(all_clip_sum, ))

        # 预测原始特征位置
        original_predicted_order = self.frame_order_prediction_layers(original_encoded_vid_proposal_feat)  # [bs, 32, 32]
        original_prediction_loss = F.cross_entropy(original_predicted_order.view(all_clip_sum, -1), original_idx.reshape(all_clip_sum, ))

        return (shuffle_prediction_loss + original_prediction_loss) / 2 + 5 * reconstruct_loss, shuffle_clip_video_feat
