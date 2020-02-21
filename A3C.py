import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import Transformer, PositionalEncoding
import numpy as np

from parameters import VOCAB_SIZE, MAX_ANSWER_SIZE, MAX_QUESTION_SIZE


class Policy_Network(nn.Module):
    def __init__(self, num_layers = 6, num_heads = 8, key_dimension = 64, 
                 value_dimension = 64, dropout = 0.1, n_position = 160, 
                 d_char_vec = 512, inner_dimension = 2048, 
                 n_trg_position = MAX_ANSWER_SIZE, n_src_position = MAX_QUESTION_SIZE, padding = 1,
                critic_num_layers=4, critic_kernel_size=4, critic_padding=1, model=None, share_embedding_layers=False):
        
        super(Policy_Network, self).__init__()
        
        self.action_transformer = Transformer(n_src_vocab=VOCAB_SIZE + 1, n_trg_vocab=VOCAB_SIZE+1, src_pad_idx=0, trg_pad_idx=0,
                               d_char_vec=d_char_vec, d_model=d_char_vec, d_inner=inner_dimension, n_layers=num_layers,
                               n_head=num_heads, d_k=key_dimension, d_v=value_dimension, dropout=dropout,
                               n_trg_position=n_trg_position, n_src_position=n_src_position,
                               trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True) if model == None else model
        
        
        critic_src_embedding = self.action_transformer.encoder.src_word_emb if share_embedding_layers else nn.Embedding(VOCAB_SIZE + 1, d_char_vec, padding_idx=self.action_transformer.src_pad_idx)
        critic_trg_embedding = self.action_transformer.deocder.trg_word_emb if share_embedding_layers else nn.Embedding(VOCAB_SIZE + 1, d_char_vec, padding_idx=self.action_transformer.trg_pad_idx)
        critic_src_position = self.action_transformer.encoder.position_enc if share_embedding_layers else PositionalEncoding(d_char_vec, n_src_position)
        critic_trg_position = self.action_transformer.decoder.position_enc if share_embedding_layers else PositionalEncoding(d_char_vec, n_trg_position)

        self.value_head = Critic(conv_layers=critic_num_layers, d_char_vec=d_char_vec, kernel_size=critic_kernel_size,
                                n_vocab=VOCAB_SIZE+1, dropout=dropout, padding=critic_padding, 
                                src_embedding=critic_src_embedding, 
                                trg_embedding=critic_trg_embedding, 
                                 src_position_enc=critic_src_position, 
                                 trg_position_enc=critic_trg_position)
        
    def forward(self, src_seq, trg_seq):
        
        action_prob = self.action_transformer(src_seq, trg_seq)
        action_prob = action_prob[:, -1, :]
        state_values = self.value_head(src_seq, trg_seq)
        
        return action_prob, state_values


class Critic(nn.Module):
    
    def __init__(self, conv_layers=4, d_char_vec=512, n_vocab=96, kernel_size=4, dropout=0.1, padding=1, 
                 src_embedding=None, trg_embedding=None, src_position_enc=None, trg_position_enc=None, pad_idx=0):
        
        super(Critic, self).__init__()
        
        self.src_word_emb = src_embedding if src_embedding != None else nn.Embedding(n_vocab, d_char_vec, padding_idx=pad_idx)
        self.trg_word_emb = trg_embedding if trg_embedding != None else nn.Embedding(n_vocab, d_char_vec, padding_idx=pad_idx)
        
        self.src_position_enc = src_position_enc if src_position_enc != None else PositionalEncoding(d_char_vec, n_position=MAX_QUESTION_SIZE)
        self.trg_position_enc = trg_position_enc if trg_position_enc != None else PositionalEncoding(d_char_vec, n_position=MAX_ANSWER_SIZE)
        self.dropout = nn.Dropout(p=dropout)
        
        self.conv_layers = [nn.Conv1d(in_channels=d_char_vec, out_channels=d_char_vec, kernel_size=kernel_size, padding=padding) for _ in range(conv_layers)]
        self.value_layer = nn.Linear(d_char_vec, 1)
        
    def forward(self, src_seq, trg_seq):
        
        batch_sz, max_src_pos, max_trg_pos = src_seq.shape[0], src_seq.shape[1], trg_seq.shape[1]
        src_seq_pad = torch.cat((src_seq, torch.zeros((batch_sz, MAX_QUESTION_SIZE-max_src_pos), dtype=torch.int64)), dim=1)
        trg_seq_pad = torch.cat((trg_seq, torch.zeros((batch_sz, MAX_ANSWER_SIZE-max_trg_pos), dtype=torch.int64)), dim=1)
        src_emb = self.dropout(self.src_position_enc(self.src_word_emb(src_seq_pad)))
        trg_emb = self.dropout(self.trg_position_enc(self.trg_word_emb(trg_seq_pad)))
        
        x = torch.cat((src_emb, trg_emb), dim=1)
        conv_out = torch.transpose(x, 1, 2).contiguous()
        for conv in self.conv_layers:
            conv_out = conv(conv_out)
        conv_out = torch.mean(F.relu(conv_out), dim=2)
        output = self.value_layer(conv_out)
        
        return output