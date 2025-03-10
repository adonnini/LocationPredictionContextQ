import torch.nn as nn
import numpy as np
import torch, math
from torch import Tensor


from models.embed import AllEmbedding
from models.fc import FullyConnected


class TransEncoder(nn.Module):
    def __init__(self, config, total_loc_num) -> None:
        super(TransEncoder, self).__init__()

        self.d_input = config['base_emb_size']
        # self.d_input = config.base_emb_size
        self.Embedding = AllEmbedding(self.d_input, config, total_loc_num)

        # encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            self.d_input, nhead=config.nhead, activation="gelu", dim_feedforward=config.dim_feedforward
        )
        encoder_norm = torch.nn.LayerNorm(self.d_input)
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_encoder_layers,
            norm=encoder_norm
        )

        self.FC = FullyConnected(self.d_input, config, if_residual_layer=True, total_loc_num=total_loc_num)

        # init parameter
        self._init_weights()

    # def forward(self, src, context_dict, device) -> Tensor:
    # def forward(self, src, context_dict, is_causal=False) -> Tensor:
    # def forward(self, src, context_dict) -> Tensor:

    def forward(self, src, context_dict, is_causal=True) -> Tensor:
        # print("LocationPrediction - models - MHSA - about to run AllEmbedding --- ")
        # emb = self.Embedding(src, context_dict)
        emb = self.Embedding(src, context_dict)
        seq_len = context_dict["len"]

        # positional encoding, dropout performed inside
        src_mask = self._generate_square_subsequent_mask(src.shape[0])
        src_padding_mask = (src == 0).transpose(0, 1)
        # src_mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
        # src_padding_mask = (src == 0).transpose(0, 1).to(device)
        out = self.encoder(emb, mask=src_mask, src_key_padding_mask=src_padding_mask, is_causal=True)
        # out = self.encoder(emb, mask=src_mask, src_key_padding_mask=src_padding_mask)

        # only take the last timestep
        out = out.gather(
            0,
            seq_len.view([1, -1, 1]).expand([1, out.shape[1], out.shape[-1]]) - 1,
        ).squeeze(0)

        return self.FC(out, context_dict["user"])

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)

    def _init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    @torch.no_grad()
    def get_attention_maps(self, x, context_dict, device):
        emb = self.Embedding(x, context_dict)
        seq_len = context_dict["len"]

        # positional encoding, dropout performed inside
        src_mask = self._generate_square_subsequent_mask(x.shape[0]).to(device)
        src_padding_mask = (x == 0).transpose(0, 1).to(device)

        attention_maps = []
        for layer in self.encoder.layers:

            _, attn_map = layer.self_attn(
                emb, emb, emb, attn_mask=src_mask, key_padding_mask=src_padding_mask, need_weights=True
            )
            # only take the last timestep
            attn_map = attn_map.gather(
                1, seq_len.view([-1, 1, 1]).expand([attn_map.shape[0], 1, attn_map.shape[-1]]) - 1
            ).squeeze(1)
            attention_maps.append(attn_map)
            emb = layer(emb)

        return attention_maps
