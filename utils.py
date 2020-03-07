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

def save_checkpoint(self, epoch, model, optimizer, scheduler, suffix="default"):
    if scheduler:
      torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
        }, "checkpoint-{}.pth".format(suffix))
    else:
      torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
        }, "checkpoint-{}.pth".format(suffix))

def from_checkpoint_if_exists(self, model, optimizer, scheduler):
    epoch = 0
    if os.path.isfile("checkpoint.pth"):
        print("Loading existing checkpoint...")
        checkpoint = torch.load("checkpoint.pth")
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
    return epoch, model, optimizer, scheduler

def tb_mle_meta_batch(tb, loss, acc, num_iter):
    tb.add_scalars(
        {
            "loss_meta_batch": loss,
            "accuracy_meta_batch": acc,
        },
        group="meta_mle_train",
        sub_group="batch",
        global_step=num_iter
    )