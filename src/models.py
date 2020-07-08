# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
import torch.nn.functional as F

from .utils import load_embeddings, normalize_embeddings


class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.emb_dim = params.emb_dim
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)


class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.emb_dim = params.emb_dim
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)

def build_model(params, with_dis):
    """
    Build all components of the model.
    """
    # source embeddings
    src_dico, _src_emb = load_embeddings(params, source=True)
    params.src_dico = src_dico
    src_emb = nn.Embedding(len(src_dico), params.emb_dim, sparse=True)
    src_emb.weight.data.copy_(_src_emb)

    # target embeddings
    if params.tgt_lang:
        tgt_dico, _tgt_emb = load_embeddings(params, source=False)
        params.tgt_dico = tgt_dico
        tgt_emb = nn.Embedding(len(tgt_dico), params.emb_dim, sparse=True)
        tgt_emb.weight.data.copy_(_tgt_emb)
    else:
        tgt_emb = None

    # mapping
    mapping = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
    if getattr(params, 'map_id_init', True):
        mapping.weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))

    # discriminator
    discriminator = Discriminator(params) if with_dis else None

    # cuda
    if params.cuda:
        src_emb.cuda()
        if params.tgt_lang:
            tgt_emb.cuda()
        mapping.cuda()
        if with_dis:
            discriminator.cuda()

    # normalize embeddings
    params.src_mean = normalize_embeddings(src_emb.weight.data, params.normalize_embeddings)
    if params.tgt_lang:
        params.tgt_mean = normalize_embeddings(tgt_emb.weight.data, params.normalize_embeddings)

    return src_emb, tgt_emb, mapping, discriminator

# BDMA.
class BiFNN(nn.Module):

    def __init__(self, params):
        super(BiFNN, self).__init__()

        self.emb_dim = params.emb_dim
        self.n_layers = params.n_layers
        self.n_hid_dim = params.n_hid_dim
        self.n_dropout = params.n_dropout
        self.n_input_dropout = params.n_input_dropout
        self.bidirectional = params.bidirectional
        self.shared_params = params.shared

        # NOTE: The output dim needs to be changed to the size of the output embedding.
        layers = []
        reverse = []

        for i in range(self.n_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.n_hid_dim
            output_dim = self.emb_dim if i == self.n_layers else self.n_hid_dim
            layers.append(nn.Linear(input_dim, output_dim, bias=False))
            reverse.insert(0, nn.Linear(output_dim, input_dim, bias=False))

            if self.shared_params:
                weights = nn.Parameter(layers[-1].weight.data)
                layers[-1].weight.data = weights.clone()
                reverse[0].weight.data = layers[-1].weight.data.t()

            if i < self.n_layers:
                layers.append(nn.LeakyReLU(0.1))
                if self.n_dropout > 0.0:
                    layers.append(nn.Dropout(self.n_dropout))
                    reverse.insert(0, nn.Dropout(self.n_dropout))

                reverse.insert(0, nn.LeakyReLU(0.1))

        # A reverse comparison for baseline is not necessarily apples-to-apples.
        self.layers = nn.Sequential(*layers)
        self.reverse = nn.Sequential(*reverse)

        print("---Network Structure---")
        print(self.layers)
        print(self.reverse)
        print("---Network Structure---")

    def forward(self, x, fdir=True):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        if self.bidirectional is True and fdir is False:
            return self.reverse(x)
        else:
            return self.layers(x)

# Build the models for BDMA.
def build_bdma_model(params, with_dis):
    """
    Build all components of the model.
    """
    # source embeddings
    src_dico, _src_emb = load_embeddings(params, source=True)
    params.src_dico = src_dico
    src_emb = nn.Embedding(len(src_dico), params.emb_dim, sparse=True)
    src_emb.weight.data.copy_(_src_emb)

    # target embeddings
    if params.tgt_lang:
        tgt_dico, _tgt_emb = load_embeddings(params, source=False)
        params.tgt_dico = tgt_dico
        tgt_emb = nn.Embedding(len(tgt_dico), params.emb_dim, sparse=True)
        tgt_emb.weight.data.copy_(_tgt_emb)
    else:
        tgt_emb = None

    # Mapping
    mapping = BiFNN(params)

    # if getattr(params, 'map_id_init', True):
    #     mapping.weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))

    # discriminator
    # discriminator = Discriminator(params) if with_dis else None

    # cuda
    if params.cuda:
        src_emb.cuda()
        if params.tgt_lang:
            tgt_emb.cuda()
        mapping.cuda()
        # if with_dis:
        #     discriminator.cuda()

    # normalize embeddings
    params.src_mean = normalize_embeddings(src_emb.weight.data, params.normalize_embeddings)
    if params.tgt_lang:
        params.tgt_mean = normalize_embeddings(tgt_emb.weight.data, params.normalize_embeddings)

    return src_emb, tgt_emb, mapping
