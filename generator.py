'''
This module will handle the text generation with beam search.
It is just a customized version of Translator.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import collections
import numpy as np
from transformer.Beam import Beam
from utils import get_subsequent_mask, get_pad_mask, np_encode_string, np_decode_string, question_to_batch_collate_fn, question_answer_to_batch_collate_fn
from dataset import PAD, np_encode_string, np_decode_string
from parameters import MAX_QUESTION_SIZE, MAX_ANSWER_SIZE, VOCAB_SIZE

worst_k = collections.deque(maxlen=10)

def build_worst_k(batch_idx, all_hyp, all_scores):
    for i, idx_seqs in enumerate(all_hyp):
        r = math_dataset.np_decode_string(np.array(idx_seqs[0]))
        s = all_scores[i][0].cpu().item()
        if len(worst_k) == 0 or s < worst_k[0][2]:
            worst_k.appendleft((batch_idx + i, r, s))

def predict(generator, data, device, max_predictions=None):
    if max_predictions is not None:
        cur = max_predictions
    else:
        cur = len(data)
            
    resps = []
    for batch_idx, batch in enumerate(data):
        if cur == 0:
            break
        

        batch_qs = batch.to(device)

        all_hyp, all_scores = generator.generate_batch(batch_qs)
        
        for i, idx_seqs in enumerate(all_hyp):
            for j, idx_seq in enumerate(idx_seqs):
                r = np_decode_string(np.array(idx_seq))
                s = all_scores[i][j].cpu().item()
                resps.append({"resp":r, "score":s})
        cur -= 1
                    
    return resps

def predict_dataset(dataset, model, device, callback, max_batches=None,
                    beam_size=5, max_token_seq_len=MAX_ANSWER_SIZE, n_best=1,
                    batch_size=1, num_workers=1):
    
    generator = Generator(model, device, beam_size=beam_size,
                          max_token_seq_len=max_token_seq_len, n_best=n_best)

    if max_batches is not None:
        cur = max_batches
    else:
        cur = len(dataset)
            
    resps = []
    for batch_idx, batch in enumerate(dataset):
        if cur == 0:
            break
        
        batch_qs, _ = map(lambda x: x.to(device), batch)
        all_hyp, all_scores = generator.generate_batch(batch_qs)
        
        callback(batch_idx, all_hyp, all_scores)
        
        cur -= 1
    return resps
    
    
def predict_multiple(questions, model, device='cpu', beam_size=5,
                     max_token_seq_len=MAX_ANSWER_SIZE, n_best=1, batch_size=1,
                     num_workers=1):

    questions = list(map(lambda q: np_encode_string(q), questions))
    questions = data.DataLoader(questions, batch_size=1, shuffle=False, collate_fn=question_to_batch_collate_fn)#, num_workers=1)
    
    generator = Generator(model, device, beam_size=beam_size, max_token_seq_len=max_token_seq_len, n_best=n_best)
        
    return predict(generator, questions, device)
    
    
def predict_single(question, model, device='cpu', beam_size=5,
                   max_token_seq_len=MAX_ANSWER_SIZE, n_best=1):
    
    generator = Generator(model, device, beam_size=beam_size,
                          max_token_seq_len=max_token_seq_len, n_best=n_best)
    
    qs = [np_encode_string(question)]
    # qs = qs.to(device)

    # max_q_len = max(len(q) for q in qs)

    # batch_qs = []
    # for q in qs:
        # batch_qs.append(np.pad(q, (0, max_q_len - len(q)), mode='constant', constant_values=PAD))

    # batch_qs = torch.LongTensor(batch_qs)
    batch_qs = torch.LongTensor(qs)

    all_hyp, all_scores = generator.generate_batch(batch_qs)
    resp = np_decode_string(np.array(all_hyp[0][0]))
    
    resps = []
    for i, idx_seqs in enumerate(all_hyp):
        for j, idx_seq in enumerate(idx_seqs):
            r = np_decode_string(np.array(idx_seq))
            s = all_scores[i][j].cpu().item()
            resps.append({"resp":r, "score":s})
    
    return resps


class Generator(object):
    ''' Load with Transformer trained model and handle the beam search '''

    def __init__(self, model, device, beam_size, n_best, max_token_seq_len=MAX_ANSWER_SIZE):
        self.model = model
        self.device = device
        self.beam_size = beam_size
        self.n_best = n_best
        self.max_token_seq_len = max_token_seq_len

        model.word_prob_prj = nn.LogSoftmax(dim=1)

        model = model.to(self.device)
        self.model.eval()

    def generate_batch(self, src_seq):
        ''' Translation work in one batch '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(
                src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(
                inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def predict_word(dec_seq, src_seq, enc_output, n_active_inst, n_bm):
                src_mask = get_pad_mask(src_seq, PAD)
                dec_mask = get_pad_mask(dec_seq, PAD) & get_subsequent_mask(dec_seq)
                dec_output, *_ = self.model.decoder(dec_seq, dec_mask, enc_output, src_mask)
                                                    
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(self.model.trg_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)

                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            # dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            word_prob = predict_word(dec_seq, src_seq, enc_output, n_active_inst, n_bm)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            #-- Encode
            # src_seq, src_pos = src_seq.to(self.device), src_pos.to(self.device)
            # src_enc, *_ = self.model.encoder(src_seq, src_pos)
            src_mask = get_pad_mask(src_seq, PAD)
            # trg_mask = get_pad_mask(trg_seq, PAD) & get_subsequent_mask(trg_seq)

            src_seq = src_seq.to(self.device)
            src_enc, *_ = self.model.encoder(src_seq, src_mask)

            #-- Repeat data for beam search
            n_bm = self.beam_size
            n_inst, len_s, d_h = src_enc.size()
            src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)

            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for len_dec_seq in range(1, self.max_token_seq_len + 1):

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, n_bm)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                    src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, self.n_best)

        return batch_hyp, batch_scores
