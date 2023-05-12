from model import TIPCB
import argparse
import torch
from transformers import AutoTokenizer
from dataset import split, TIPCB_data, NPZ_data
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_dir', type=str,
                    default="./checkpoint",
                    help='directory to store checkpoint')
parser.add_argument('--log_dir', type=str,
                    default="./log",
                    help='directory to store log')

#word_embedding
parser.add_argument('--max_length', type=int, default=64)

#image setting
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--height', type=int, default=384)
parser.add_argument('--feature_size', type=int, default=2048)

#experiment setting
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epoches', type=int, default=100)

#loss function setting
parser.add_argument('--epsilon', type=float, default=1e-8)

# the root of the data folder
parser.add_argument("--image_root_path", type=str, default='/home/palm/PycharmProjects/text_image_retrieval/CUHK-PEDES/imgs')

parser.add_argument('--adam_lr', type=float, default=0.003, help='the learning rate of adam')
parser.add_argument('--wd', type=float, default=0.00004)
parser.add_argument('--lr_decay_type', type=str, default='MultiStepLR',
                        help='One of "MultiStepLR" or "StepLR" or "ReduceLROnPlateau"')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
parser.add_argument('--epoches_decay', type=str, default='40', help='#epoches when learning rate decays')
parser.add_argument('--warm_epoch', default=10, type=int, help='the first K epoch that needs warm up')

parser.add_argument("--language", default="en", type=str, help="the language to train for")

args = parser.parse_args()



# ------------------------ test dataset ------------------------
# with open("/aicity/TIPCB/data/BERT_en_original/BERT_id_train_64_new.npz", "rb") as f:
#     train = pickle.load(f)

# with open("/aicity/TIPCB/data/BERT_en_original/BERT_id_val_64_new.npz", "rb") as f:
#     val = pickle.load(f)

# train_dataset = NPZ_data(train, args)
# val_dataset = NPZ_data(val, args, train=False)
# train_dl = DataLoader(train_dataset, batch_size=2)
# val_dl = DataLoader(val_dataset, batch_size=2)

# for batch in val_dl:
#     print(batch)
#     break

# ----------------------------------------------------------------




early_stopping = EarlyStopping('val_rank1', mode='max', patience=10)
checkpoint_callback = ModelCheckpoint(
    dirpath=args.checkpoint_dir,
    filename='{epoch}-{val_rank1:.4f}-{val_loss:.4f}',
    monitor='val_rank1',
    mode='max'
)

pl.seed_everything(0)
tb_logger = pl_loggers.TensorBoardLogger(args.log_dir)

if args.language == "en":
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2', model_max_length=args.max_length)
    train_list, val_list = split('/home/palm/PycharmProjects/text_image_retrieval/CUHK-PEDES/caption_all.json')
    train_dataset = TIPCB_data(train_list, tokenizer, args)
    val_dataset = TIPCB_data(val_list, tokenizer, args, train=False)
# elif # removed for Thai language

# with open("/aicity/TIPCB/data/BERT_en_original/BERT_id_train_64_new.npz", "rb") as f:
#     train = pickle.load(f)

# with open("/aicity/TIPCB/data/BERT_en_original/BERT_id_test_64_new.npz", "rb") as f:
#     val = pickle.load(f)

# train_dataset = NPZ_data(train, args)
# val_dataset = NPZ_data(val, args, train=False)
train_dl = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)

model = TIPCB(args, val_len=len(val_dl))

trainer = pl.Trainer(amp_level='O1', amp_backend="apex",
                        max_epochs=args.num_epoches,
                        callbacks=[checkpoint_callback],
                        gpus=1,
                        accumulate_grad_batches=1,
                        logger=tb_logger)
trainer.fit(model, train_dl, val_dl)




# ----------------------------------------------------------------