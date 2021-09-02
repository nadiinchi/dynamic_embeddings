from model import MixtureAttention
from data import DataLoader
import os
import sys
from tqdm import tqdm
from utils import adjust_learning_rate, compute_acc, VocabManager
import torch
import argparse

import logger

parser = argparse.ArgumentParser(description='Training model.')
parser.add_argument('--dir', type=str, default='logs/', metavar='DIR',
                    help='where to save logs')
parser.add_argument('--comment', type=str, default="", metavar='T', help='comment                         to the experiment')
parser.add_argument('--wd', type=float, default=1e-2, metavar='WD',
                help='weight decay')
parser.add_argument('--N_filename', type=str, \
                    default='./pickle_data/PY_non_terminal.pickle', \
                    metavar='DIR',
                    help='N_filename')
parser.add_argument('--T_filename', type=str, \
                   default='./pickle_data/PY_terminal.pickle',\
                    metavar='DIR',
                    help='T_filename')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--sieve_step', type=int, default=0, help="if >0, every sieve_step-s data object is included in the training data, others are dropped")
parser.add_argument('--loss_type', type=int, default=1, help="see help in utils.py/class VocPointerLoss")
parser.add_argument('--num_epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--lr_decay', type=float, default=0.6, metavar='M',
                    help='lr decay (applied after each epoch)')
parser.add_argument('--opt', type=str, default="adamw", metavar='S',
                    help='optimizer: adamw or adam')
parser.add_argument('--grad_clip', type=float, default=5, metavar='WD',
                    help='max norm to clip gradient')
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--hidden_size', type=int, default=1500)
parser.add_argument('--vocab_size', type=int, default=50000)
parser.add_argument('--vocab_size_out', type=int, default=None)
parser.add_argument('--embedding_sizeT', type=int, default=1200)
parser.add_argument('--embedding_sizeT_dyn', type=int, default=None, help="used only in mixed_uni")
parser.add_argument('--embedding_sizeN', type=int, default=300)
parser.add_argument('--truncate_size', type=int, default=50)
parser.add_argument('--no_pointer', action='store_true')
parser.add_argument('--no_attn', action='store_true')
parser.add_argument("--tie_weights", action="store_true")
parser.add_argument('--anonym', default=None, \
                    help="not None => turn on anonimized model, mixed (init dyn emb independently uniformly) | pure_uni (init dyn emb similarly uniformly) | pure_zero (init dyn emb similarly with zeros)")
parser.add_argument('--static_vocab_size', type=int, default=1,\
                   help="if anonym is not None, sets the first N tokens in the vocabulary to be static. Default: 1 (static embeddings only for EMPTY token)")
parser.add_argument('--no_train_init', action='store_true')
parser.add_argument('--save_fq', type=int, default=50, metavar='N',
                    help='save frequency (default: 50)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--not_save_weights', action='store_true',
                    help='not save weights')
parser.add_argument('--no_verbose', action='store_true',
                    help='turn off printing loss each 100it')
parser.add_argument('--test_code', action='store_true')
parser.add_argument('--save_verbose', action='store_true')
args = parser.parse_args()

if args.test_code:
    args.not_save_weights = True
    args.comment = "test_code"

fmt_list = [('lr', "3.4e"), ('tr_loss', "3.3e"), \
            ('tr_acc', '9.4f'), \
        ('test_loss', "3.3e"), ('test_acc', '9.4f')]
fmt = dict(fmt_list)
log = logger.Logger(args.comment, fmt=fmt, base=args.dir)
log.print(" ".join(sys.argv))
log.print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader = DataLoader(args.N_filename,
            args.T_filename,
            True,
            args.truncate_size,
            args.batch_size,
            sieve_step=args.sieve_step)

test_dataloader = DataLoader(args.N_filename,
            args.T_filename,
            False,
            args.truncate_size,
            args.batch_size)
log.print("Training data: %d tokens in total, testing data: %d tokens in total"%(train_dataloader.split_N.size, test_dataloader.split_N.size))

vocab_manager = VocabManager(args.vocab_size, args.vocab_size_out, args.anonym, \
                             static_vocab_size=args.static_vocab_size)

if args.anonym is not None:
    static_x_ids = vocab_manager.static_x_ids_in
    hidden_size = args.hidden_size
else:
    static_x_ids = []
    hidden_size = args.hidden_size

model = MixtureAttention(
        hidden_size = hidden_size,
        vocab_sizeT_in = vocab_manager.vocab_size_in,
        vocab_sizeT_out = vocab_manager.vocab_size_out,
        vocab_sizeN = train_dataloader.vocab_sizeN,
        embedding_sizeT = args.embedding_sizeT,
        embedding_sizeN = args.embedding_sizeN,
        dropout = args.dropout,
        attn = not args.no_attn,
        pointer = not args.no_pointer,
        train_init = not args.no_train_init,
        anonymized=args.anonym,
        static_x_ids=static_x_ids,
        attn_size=train_dataloader.attn_size,
        eof_N_id=train_dataloader.eof_idxN,\
        unk_id_out=vocab_manager.unk_idx_out,\
        loss_type=args.loss_type,
        tie_weights = args.tie_weights,
        embedding_sizeT_dyn=args.embedding_sizeT_dyn
    ).to(device)

for name, p in model.named_parameters():
    if "weight" in name:
        p.data.uniform_(-0.05, 0.05)

start_epoch = 1
if args.resume is not None:
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch'] + 1
    args.lr = args.lr * args.decay_ratio ** (start_epoch-1)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

Opt = torch.optim.Adam if args.opt=="adam" else torch.optim.AdamW
optimizer = Opt(model.parameters(), lr=args.lr, weight_decay=args.wd)

def train():
    best_acc = 0
    for epoch in range(start_epoch, args.num_epochs+1):
        if args.lr_decay < 1:
            lr = args.lr * args.lr_decay ** max(epoch - 1, 0)
            adjust_learning_rate(optimizer, lr)
        else:
            lr = args.lr

        loss_avg, acc_avg = 0, 0
        total = len(train_dataloader)
        
        batch_N_prev = torch.zeros((args.batch_size, 1)).long().to(device) + \
                 train_dataloader.eof_idxN
        batch_T_voc_prev = torch.zeros((args.batch_size, 1)).long().to(device) +\
                         vocab_manager.eof_idxT_in
        batch_P_prev = torch.zeros((args.batch_size, 1)).long().to(device)
        hs_prev = torch.zeros((args.batch_size, 1, model.hidden_size)).to(device)
        hc_prev = [torch.zeros((1, 1, 1)).to(device),\
                   torch.zeros((1, 1, 1)).to(device)]
        if args.anonym is not None:
            dyn_embs = [torch.zeros((args.batch_size, \
                       model.decoder.vocab_size_dyn)).long().to(device)-1,\
            torch.zeros((0, 1, \
                         model.decoder.embedding_sizeT_dyn)).float().to(device),\
            torch.zeros((0, 1, \
                         model.decoder.embedding_sizeT_dyn)).float().to(device)]
        batch_T_raw_prev = batch_T_voc_prev

        model.train()
        vocab_manager.reset()
        for i, batch in enumerate(tqdm(train_dataloader)):
            batch_N, batch_T_raw, batch_T_attn, batch_P = batch
            batch_N, batch_T_raw, batch_T_attn, batch_P = \
                                                     batch_N.to(device), \
                                                     batch_T_raw.to(device), \
                                                     batch_T_attn.to(device), \
                                                     batch_P.to(device)
            batch_T_voc_in, batch_T_voc_out = vocab_manager.get_batch_T_voc(batch_T_raw)
            optimizer.zero_grad()
            
            forward_result = model(batch_N, batch_T_voc_in, batch_T_voc_out, \
                                   batch_T_attn, batch_P, \
                                   batch_N_prev, batch_T_voc_prev, batch_P_prev,\
                                   hs_prev, hc_prev, \
                                   dyn_embs if args.anonym else None)
            if args.anonym is None:
                loss, ans, hs_, hc_ = forward_result
            else:
                loss, ans, hs_, hc_, dyn_embs = forward_result
            loss_avg += loss.item()
            
            acc = compute_acc(ans, batch_T_raw, batch_T_raw_prev, \
                                    vocab_manager.vocab_size_out,
                                    vocab_manager.vocab_size_out_cl,\
                                    vocab_manager.unk_idx_out,\
                                    vocab_manager.eof_idxT_out)
            acc_avg += acc.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), \
                                           args.grad_clip)
            optimizer.step()
            
            batch_N_prev = batch_N
            batch_T_voc_prev = batch_T_voc_in
            batch_P_prev = batch_P
            batch_T_raw_prev = batch_T_raw
            hs_prev = hs_
            hc_prev = hc_

            if (i + 1) % 100 == 0 and not args.no_verbose:
                prints = 'temp_loss: %f, temp_acc: %f' % (loss_avg/(i+1), acc_avg/(i+1))
                if args.save_verbose:
                    log.print(prints)
                else:
                    print(prints, flush=True)
            
            if args.test_code and i > 3:
                break
        
        train_loss = loss_avg / total
        train_acc = acc_avg / total

        batch_N_prev = torch.zeros((args.batch_size, 1)).long().to(device) + \
                 train_dataloader.eof_idxN
        batch_T_voc_prev = torch.zeros((args.batch_size, 1)).long().to(device) +\
                         vocab_manager.eof_idxT_in
        batch_P_prev = torch.zeros((args.batch_size, 1)).long().to(device)
        hs_prev = torch.zeros((args.batch_size, 1, model.hidden_size)).to(device)
        hc_prev = [torch.zeros((1, 1, 1)).to(device),\
                   torch.zeros((1, 1, 1)).to(device)]
        if args.anonym is not None:
            dyn_embs = [torch.zeros((args.batch_size, \
                        model.decoder.vocab_size_dyn)).long().to(device)-1,\
            torch.zeros((0, 1, \
                        model.decoder.embedding_sizeT_dyn)).float().to(device),\
            torch.zeros((0, 1, \
                        model.decoder.embedding_sizeT_dyn)).float().to(device)]
        batch_T_raw_prev = batch_T_voc_prev
        with torch.no_grad():
            model.eval()
            total_acc = 0.
            loss_eval = 0.
            vocab_manager.reset()
            for i, batch in enumerate(tqdm(test_dataloader)):
                batch_N, batch_T_raw, batch_T_attn, batch_P = batch
                batch_N, batch_T_raw, batch_T_attn, batch_P = \
                                                     batch_N.to(device), \
                                                     batch_T_raw.to(device), \
                                                     batch_T_attn.to(device), \
                                                     batch_P.to(device)
                batch_T_voc_in, batch_T_voc_out = vocab_manager.get_batch_T_voc(batch_T_raw)
                forward_result = model(batch_N, batch_T_voc_in, batch_T_voc_out, \
                                   batch_T_attn, batch_P, \
                                   batch_N_prev, batch_T_voc_prev, batch_P_prev,\
                                   hs_prev, hc_prev,\
                                   dyn_embs if args.anonym else None)
                if args.anonym is None:
                    loss, ans, hs_, hc_ = forward_result
                else:
                    loss, ans, hs_, hc_, dyn_embs = forward_result
                loss_eval += loss.item()
                acc = compute_acc(ans, batch_T_raw, batch_T_raw_prev, \
                                    vocab_manager.vocab_size_out,
                                    vocab_manager.vocab_size_out_cl,\
                                    vocab_manager.unk_idx_out,\
                                    vocab_manager.eof_idxT_out)
                total_acc += acc.item()
                batch_N_prev = batch_N
                batch_T_voc_prev = batch_T_voc_in
                batch_P_prev = batch_P
                batch_T_raw_prev = batch_T_raw
                hs_prev = hs_
                hc_prev = hc_
                
                if args.test_code:
                    break
                
            total_acc /= len(test_dataloader)
            loss_eval /= len(test_dataloader)
        values = [lr, train_loss, train_acc, \
                  loss_eval, total_acc]
        for (k, _), v in zip(fmt_list, values):
            log.add(epoch, **{k:v})

        log.iter_info()
        log.save(silent=True)
        
        if not args.not_save_weights:
            if (epoch + 1) % args.save_fq == 0:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }, log.path+'/model_epoch_%04d.cpt'%(epoch))
            elif total_acc > best_acc:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }, log.path+'/best_model.cpt')
        if total_acc > best_acc:
            best_acc = total_acc
        if args.test_code:
            break

if __name__ == '__main__': 
    train()
