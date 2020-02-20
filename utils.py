import torch
import numpy as np
from transformer.Constants import PAD, UNK, BOS, EOS

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

def question_to_batch_collate_fn(qs):
    ''' Gather + Pad the question to the max seq length in batch '''

    max_q_len = max(len(q) for q in qs)

    batch_qs = []
    for q in qs:
        batch_qs.append(np.pad(q, (0, max_q_len - len(q)), mode='constant', constant_values=PAD))
    
    batch_qs = torch.LongTensor(batch_qs)

    return batch_qs

def question_answer_to_batch_collate_fn(qas):
    ''' Gather + Pad the question/answer to the max seq length in batch '''

    max_q_len = max(len(qa["q_enc"]) for qa in qas)
    max_a_len = max(len(qa["a_enc"]) for qa in qas)

    batch_qs = []
    batch_as = []
    batch_pos = []
    for qa in qas:
      batch_qs.append(np.pad(qa["q_enc"], (0, max_q_len - len(qa["q_enc"])), mode='constant', constant_values=PAD))
      batch_as.append(np.pad(qa["a_enc"], (0, max_a_len - len(qa["a_enc"])), mode='constant', constant_values=PAD))
    
    batch_qs = torch.LongTensor(batch_qs)
    batch_as = torch.LongTensor(batch_as)

    return batch_qs, batch_as

def np_encode_string(s, char0 = ord(' ')):
    """converts a string into a numpy array of bytes
    (char0 - 1) is subtracted from all bytes values (0 is used for PAD)
    string is pre-pended with BOS and post-pended with EOS"""
    chars = np.array(list(s), dtype='S1').view(np.uint8)
    # normalize to 1 - 96, 0 being PAD
    chars = chars - char0 + 1

    chars = np.insert(chars, 0, BOS)
    chars = np.insert(chars, len(chars), EOS)
    return chars

def np_decode_string(chars, char0 = ord(' ')):
    """converts a numpy array of bytes into a UTF-8 string
    (char0 - 1) is added to all bytes values (0 is used for PAD)
    BOS/EOS are removed before utf-8 decoding"""
    chars = chars.astype(np.uint8)
    chars = chars + char0 - 1
    chars = chars[:-1]
    chars = chars.tobytes()
    s = chars.decode('UTF-8')
    return s