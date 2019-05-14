
'''
This script handling the training process.
'''

import math
import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from dataset import TranslationDataset, paired_collate_fn
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def train_epoch(model, training_data, optimizer, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):
        
        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
        print('PREDICT:')
        print(pred.size())
        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt['log']:
        log_train_file = opt['log'] + '.train.log'
        log_valid_file = opt['log'] + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt['epoch']):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device, smoothing=opt['label_smoothing'])
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt['save_model']:
            if opt['save_mode'] == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt['save_mode'] == 'best':
                model_name = opt['save_model'] + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))


def prepare_dataloaders(data, batch_size):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['train']['src'],
            tgt_insts=data['train']['tgt']),
        num_workers=2,
        batch_size=batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']),
        num_workers=2,
        batch_size=batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader


# python train.py -data data/multi30k.atok.low.pt 
#                 -save_model trained
#                 -save_mode best
#                 -proj_share_weight
#                 -label_smoothing

opt = {'data': 'data/multi30k.atok.low.pt',
       'epoch': 10,
       'batch_size': 16,
       'd_model': 512,
       'd_inner_hid': 2048,
       'd_k': 64,
       'd_v': 64,
       'n_head': 8,
       'n_layers': 6,
       'n_warmup_steps': 4000,
       'dropout': 0.1,
       'embs_share_weight': False,
       'proj_share_weight': True,
       'log': None,
       'save_model': 'trained',
       'save_mode': 'best',
       'no_cuda': True,
       'label_smoothing': True}

opt['cuda'] = not opt['no_cuda']
opt['d_word_vec'] = opt['d_model']

#========= Loading Dataset =========#
data = torch.load(opt['data'])
opt['max_token_seq_len'] = data['settings'].max_token_seq_len

training_data, validation_data = prepare_dataloaders(data, opt['batch_size'])

opt['src_vocab_size'] = training_data.dataset.src_vocab_size
opt['tgt_vocab_size'] = training_data.dataset.tgt_vocab_size

print(opt['src_vocab_size'])
print(opt['tgt_vocab_size'])

#========= Preparing Model =========#
if opt['embs_share_weight']:
    assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx,         'The src/tgt word2idx table are different but asked to share word embedding.'


for i in training_data:
    for j in i:
        print(j.size())
        print()
    break


train_data = data['train']['tgt']
print(len(train_data))

for i in range(0, len(train_data)):
    print(train_data[i])
    if i == 10:
        break


valid_data = data['valid']['src']
print(len(valid_data))

for i in range(0, len(valid_data)):
    print(valid_data[i])
    if i == 10:
        break


dict_data = data['dict']['tgt']
print(len(dict_data))

for k, v in dict_data.items():
    print(k)
    print(v)
    break


device = torch.device('cuda' if opt['cuda'] else 'cpu')
print(device)

transformer = Transformer(
    opt['src_vocab_size'],
    opt['tgt_vocab_size'],
    opt['max_token_seq_len'],
    tgt_emb_prj_weight_sharing=opt['proj_share_weight'],
    emb_src_tgt_weight_sharing=opt['embs_share_weight'],
    d_k=opt['d_k'],
    d_v=opt['d_v'],
    d_model=opt['d_model'],
    d_word_vec=opt['d_word_vec'],
    d_inner=opt['d_inner_hid'],
    n_layers=opt['n_layers'],
    n_head=opt['n_head'],
    dropout=opt['dropout']).to(device)


for key in opt:
    print('{}: \n{}\n'.format(key, opt[key]))
    break
    
print(opt.keys())
print(opt['batch_size'])

# d_model的512应该是word embedding的长度，一个word用512个长度的vector表示


optimizer = ScheduledOptim(
    optim.Adam(
        filter(lambda x: x.requires_grad, transformer.parameters()),
        betas=(0.9, 0.98), eps=1e-09),
    opt['d_model'], opt['n_warmup_steps'])

# TODO: 研究data的输入和输出，写一个dataloader
# TODO: 写一个只包含encoder的transformer

train(transformer, training_data, validation_data, optimizer, device, opt)





