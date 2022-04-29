# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import thumt.utils as utils
import thumt.modules as modules


class DeepLSTMEncoderLayer(modules.Module):

    def __init__(self, params, name="encoder_layer"):
        super(DeepLSTMEncoderLayer, self).__init__(name)
        self.dropout = params.residual_dropout

        with utils.scope(name):
            self.lstm = modules.LSTMCell(params.hidden_size,
                                         params.hidden_size,
                                         normalization=params.lstm_normalization)
            self.ffn = modules.FeedForward(params.hidden_size,
                                           params.filter_size,
                                           dropout=params.relu_dropout)

    def forward(self, x, state=None):
        # x: [length, batch, hidden_size]
        steps = x.size(0)
        batches = x.size(1)
        
        if state is None:
            state = self.lstm.init_state(batches, dtype=x.dtype, device=x.device)

        # lstm
        hiddens = []
        for i in range(steps):
            hidden, state = self.lstm(x[i], state)

            # hidden: [batch, hidden_size] -> [1, batch, hidden_size]
            hidden = torch.unsqueeze(hidden, dim=0)
            hiddens.append(hidden)

        hiddens = torch.cat(hiddens, dim=0)

        # ffn
        hiddens = self.ffn(hiddens)
        hiddens = nn.functional.dropout(hiddens, self.dropout, self.training)

        return hiddens, state


class DeepLSTMDecoderLayer(nn.Module):

    def __init__(self, params, name="decoder_layer"):
        super(DeepLSTMDecoderLayer, self).__init__()
        self.dropout = params.residual_dropout
        self.comb_mode = params.comb_mode

        with utils.scope(name):
            self.ffn = modules.FeedForward(params.hidden_size,
                                           params.filter_size,
                                           dropout=params.relu_dropout)

            self.src_attention = modules.MultiHeadAttention(params.hidden_size,
                                                            params.num_heads,
                                                            params.attention_dropout)
            self.tgt_attention = modules.MultiHeadAttention(params.hidden_size,
                                                            params.num_heads,
                                                            params.attention_dropout)

            # x + context
            self.lstm = modules.LSTMCell(params.hidden_size + params.hidden_size,
                                         params.hidden_size,
                                         normalization=params.lstm_normalization)

    def _attention(self, query, bias, memory=None):
        # query, memory: [length, batch, hidden_size] -> [batch, length, hidden_size]
        query = torch.transpose(query, 0, 1)
        if memory is not None:
            memory = torch.transpose(memory, 0, 1)

        # c: [batch, length, hidden_size]
        c = self.src_attention(query, bias, memory)

        # c: [batch, length, hidden_size] -> [length, batch, hidden_size]
        return torch.transpose(c, 0, 1)

    def _comb(self, c_src, c_tgt, mode="sum"):
        if self.comb_mode == "sum":
            return c_src + c_tgt
        elif self.comb_mode == "gate":
            pass
        elif self.comb_mode == "hybird":
            pass

    def forward(self, x, src_bias, tgt_bias, memory=None, state=None):
        # x, memory, c: [length, batch, hidden_size]
        steps = x.size(0)
        batches = x.size(1)

        # src-attention
        c_src = self._attention(x, src_bias, memory)
        # tgt-attention
        c_tgt = self._attention(x, tgt_bias, memory=None)
        # comb
        c = self._comb(c_src, c_tgt, mode=self.comb_mode)

        if state is None:
            state = self.lstm.init_state(batches, dtype=x.dtype, device=x.device)

        # lstm
        hiddens = []
        for i in range(steps):
            # concat(ffn_output, context)
            hidden, state = self.lstm(torch.cat((x[i], c[i]), -1), state)

            # hidden: [batch, hidden_size] -> [1, batch, hidden_size]
            hidden = torch.unsqueeze(hidden, dim=0)
            hiddens.append(hidden)

        hiddens = torch.cat(hiddens, dim=0)

        # ffn
        hiddens = self.ffn(hiddens)
        hiddens = nn.functional.dropout(hiddens, self.dropout, self.training)

        return hiddens, state


class DeepLSTMEncoder(modules.Module):

    def __init__(self, params, name="encoder"):
        super(DeepLSTMEncoder, self).__init__(name)

        with utils.scope(name):
            self.layers = nn.ModuleList([
                DeepLSTMEncoderLayer(params, name="encoder_layer_%d" % i)
                for i in range(params.num_encoder_layers)])

    def forward(self, x, batch_first=False):
        if batch_first:
            # x: [batch, length, hidden_size] -> [length, batch, hidden_size]
            x = torch.transpose(x, 0, 1)

        state = None
        for i in range(len(self.layers)):
            if i % 2 == 0:
                x, state = self.layers[i](x, state)
            else:
                x, state = self.layers[i](torch.flip(x, dims=[0]), state)
                torch.flip(x, dims=[0])

        if batch_first:
            # x: [length, batch, hidden_size] -> [batch, length, hidden_size]
            x = torch.transpose(x, 0, 1)

        return x


class DeepLSTMDecoder(modules.Module):

    def __init__(self, params, name="decoder"):
        super(DeepLSTMDecoder, self).__init__(name)

        with utils.scope(name):
            self.layers = nn.ModuleList([
                DeepLSTMDecoderLayer(params, name="decoder_layer_%d" % i)
                for i in range(params.num_decoder_layers)])

    def forward(self, x, src_bias, tgt_bias, memory=None, batch_first=False):
        # x, memory: [batch, length, hidden_size] -> [length, batch, hidden_size]
        if batch_first:
            x = torch.transpose(x, 0, 1)
            if memory is not None:
                memory = torch.transpose(memory, 0, 1)

        state = None
        for i in range(len(self.layers)):
            x, state = self.layers[i](x, src_bias, tgt_bias, memory, state)

        if batch_first:
            # x: [length, batch, hidden_size] -> [batch, length, hidden_size]
            x = torch.transpose(x, 0, 1)
        return x


class DeepLSTM(modules.Module):

    def __init__(self, params, name="deep_lstm"):
        super(DeepLSTM, self).__init__(name)
        self.params = params

        with utils.scope(name):
            self.build_embedding(params)
            self.encoder = DeepLSTMEncoder(params)
            self.decoder = DeepLSTMDecoder(params)

        self.criterion = modules.SmoothedCrossEntropyLoss(params.label_smoothing)
        self.dropout = params.residual_dropout
        self.hidden_size = params.hidden_size
        self.num_encoder_layers = params.num_encoder_layers
        self.num_decoder_layers = params.num_decoder_layers

        self.reset_parameters()

    def build_embedding(self, params):
        svoc_size = len(params.vocabulary["source"])
        tvoc_size = len(params.vocabulary["target"])

        self.src_embedding = torch.nn.Parameter(
            torch.empty([svoc_size, params.hidden_size]))
        # for what?
        self.embedding_bias = torch.nn.Parameter(
            torch.zeros([params.hidden_size]))
        self.add_name(self.embedding_bias, "embedding_bias")

        self.tgt_embedding = torch.nn.Parameter(
            torch.empty([tvoc_size, params.hidden_size]))
        self.add_name(self.src_embedding, "src_embedding")
        self.add_name(self.tgt_embedding, "tgt_embedding")

        self.softmax_weights = torch.nn.Parameter(
            torch.empty([tvoc_size, params.hidden_size]))
        self.add_name(self.softmax_weights, "softmax_weights")

    def reset_parameters(self):
        nn.init.normal_(self.src_embedding, mean=0.0,
                        std=self.params.hidden_size ** -0.5)
        nn.init.normal_(self.tgt_embedding, mean=0.0,
                        std=self.params.hidden_size ** -0.5)
        nn.init.normal_(self.softmax_weights, mean=0.0,
                        std=self.params.hidden_size ** -0.5)

    def encode(self, features, state):
        src_seq = features["source"]
        src_mask = features["source_mask"]
        src_attn_bias = self.masking_bias(src_mask)

        inputs = torch.nn.functional.embedding(src_seq, self.src_embedding)
        inputs = inputs * (self.hidden_size ** 0.5)
        inputs = inputs + self.embedding_bias

        src_attn_bias = src_attn_bias.to(inputs)

        # inputs, outputs: [batch, length, hidden_size]
        encoder_output = self.encoder(inputs, batch_first=True)

        state["encoder_output"] = encoder_output
        state["src_attn_bias"] = src_attn_bias

        return state

    def decode(self, features, state, mode="infer"):
        tgt_seq = features["target"]

        src_attn_bias = state["src_attn_bias"]
        tgt_attn_bias = self.causal_bias(tgt_seq.shape[1])

        targets = torch.nn.functional.embedding(tgt_seq, self.tgt_embedding)
        targets = targets * (self.hidden_size ** 0.5)

        decoder_input = torch.cat(
            [targets.new_zeros([targets.shape[0], 1, targets.shape[-1]]),
             targets[:, 1:, :]], dim=1)

        encoder_output = state["encoder_output"]
        tgt_attn_bias = tgt_attn_bias.to(targets)

        if mode == "infer":
            decoder_input = decoder_input[:, -1:, :]
            tgt_attn_bias = tgt_attn_bias[:, :, -1:, :]

        # encoder_output, decoder_input, decoder_output: [batch, length, hidden_size]
        decoder_output = self.decoder(decoder_input, src_attn_bias, tgt_attn_bias,
                                      memory=encoder_output, batch_first=True)

        decoder_output = torch.reshape(decoder_output, [-1, self.hidden_size])
        decoder_output = torch.transpose(decoder_output, -1, -2)
        logits = torch.matmul(self.softmax_weights, decoder_output)
        logits = torch.transpose(logits, 0, 1)

        # logits: [batch * length, tvoc_size]
        return logits, state

    def forward(self, features, labels, mode="train", level="sentence"):
        mask = features["target_mask"]

        state = self.empty_state(features["target"].shape[0], labels.device)
        state = self.encode(features, state)
        logits, _ = self.decode(features, state, mode=mode)

        loss = self.criterion(logits, labels)
        mask = mask.to(torch.float32)

        # Prevent FP16 overflow
        if loss.dtype == torch.float16:
            loss = loss.to(torch.float32)

        if mode == "eval":
            if level == "sentence":
                return -torch.sum(loss * mask, 1)
            else:
                return torch.exp(-loss) * mask - (1 - mask)

        return (torch.sum(loss * mask) / torch.sum(mask)).to(logits)

    def empty_state(self, batch_size, device):
        state = {}
        return state

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = (1.0 - mask) * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1), 1)

    @staticmethod
    def causal_bias(length, inf=-1e9):
        ret = torch.ones([length, length]) * inf
        ret = torch.triu(ret, diagonal=1)
        return torch.reshape(ret, [1, 1, length, length])

    @staticmethod
    def base_params():
        params = utils.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            hidden_size=512,
            filter_size=2048,
            num_heads=1,
            num_encoder_layers=6,
            num_decoder_layers=4,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            lstm_normalization=False,
            # normalization="after",
            # shared_embedding_and_softmax_weights=False,
            # shared_source_target_embedding=False,
            # Override default parameters
            warmup_steps=4000,
            train_steps=100000,
            learning_rate=7e-4,
            learning_rate_schedule="linear_warmup_rsqrt_decay",
            batch_size=4096,
            fixed_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0,
            comb_mode="sum"
        )

        return params

    @staticmethod
    def default_params(name=None):
        if name == "base":
            return DeepLSTM.base_params()
        else:
            return DeepLSTM.base_params()
