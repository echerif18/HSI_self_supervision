import torch
import torch.nn as nn
from typing import Callable, Optional, Union
from enum import Enum
from itertools import repeat
import collections.abc
import numpy as np
from utils import *

### the code was adapted from : https://github.com/Romain3Ch216/tlse-experiments/tree/main

class MaskedAutoencoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, n_bands=310, seq_size=5, in_chans=1,
                 embed_dim=32, depth=4, num_heads=4,
                 decoder_embed_dim=32, decoder_depth=4, decoder_num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, cls_token=True):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.seq_embed = SeqEmbed(n_bands, seq_size, in_chans, embed_dim)
        num_sequences = self.seq_embed.num_sequences

        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.is_cls_token = True
        else:
            self.is_cls_token = False

        self.pos_embed = nn.Parameter(torch.zeros(1, num_sequences + np.sum(self.is_cls_token), embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        n_params = 0
        for param in self.parameters():
            n_params += param.shape.numel()
        print(f'Encoder has {n_params} parameters.')
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_sequences + np.sum(self.is_cls_token), decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, seq_size * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.initialize_weights()

        n_params = -n_params
        for param in self.parameters():
            n_params += param.shape.numel()
        print(f'Decoder has {n_params} parameters.')


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.seq_embed.num_sequences, cls_token=self.is_cls_token)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.seq_embed.num_sequences, cls_token=self.is_cls_token)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.seq_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        if self.is_cls_token:
            torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def sequencify(self, spectra):
        """
        spectra: (batch_size, n_bands)
        x: (batch_size, num_sequences, seq_size)
        """
        seq_size = self.seq_embed.seq_size
        assert spectra.shape[1] % seq_size == 0

        num_sequences = spectra.shape[1] // seq_size
        x = spectra.reshape(shape=(spectra.shape[0], num_sequences, seq_size))
        return x

    def unsequencify(self, x):
        """
        x: (batch_size, num_sequences, seq_size)
        spectra: (batch_size, n_bands)
        """
        spectra = x.reshape(shape=(x.shape[0], -1))
        return spectra

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.seq_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, np.sum(self.is_cls_token):, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        if self.is_cls_token:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, np.sum(self.is_cls_token):, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        if self.is_cls_token:
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        else:
            x = x_

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        if self.is_cls_token:
            x = x[:, 1:, :]

        return x

    def forward_loss(self, spectra, pred, mask):
        """
        spectra: [batch_size, n_bands]
        pred: [batch_size, num_sequences, seq_size]
        mask: [batch_size, num_sequences], 0 is keep, 1 is remove,
        """
        target = self.sequencify(spectra)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        if mask.sum() > 0:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        else:
            loss = loss.mean()
        return loss

    def latent(self, spectra):
        latent, _, _ = self.forward_encoder(spectra, mask_ratio=0)
        if self.is_cls_token:
            latent = latent[:, 0, :]
        else:
            latent = torch.mean(latent[:, 1:, :], dim=1)
        return latent

    def forward(self, spectra, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(spectra, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(spectra, pred, mask)
        return loss, pred, mask, latent