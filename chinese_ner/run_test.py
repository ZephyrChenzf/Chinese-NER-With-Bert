"""
@autor: chenzf
@file: run_chinese_ner.py
@time: 2019/4/9 12:59 PM

"""
import argparse
import torch
import logging
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from chinese_ner import data_process
from chinese_ner.chinese_ner_model import BertChineseNER
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from chinese_ner.getEn import getEnt


from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME

logger = logging.getLogger(__name__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def accuracy(out, labels,label_mask):
    outputs = np.argmax(out, axis=2)
    return np.sum((outputs == labels)*label_mask)

def main(args):
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(device, n_gpu, bool(args.local_rank != -1), args.fp16))
    # all_data=data_process.get_data(args.data_dir)
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    task_name = args.task_name.lower()
    print('task name: {}'.format(task_name))

    train_data_dic=torch.load('./ner_data/train_data_dic')
    train_x=train_data_dic['x']
    train_y=train_data_dic['y']
    train_mask=train_data_dic['mask']
    num_labels=train_data_dic['num_labels']
    inx2tag=train_data_dic['inx2tag']
    train_x,dev_x,train_y,dev_y,train_mask,dev_mask=train_test_split(train_x,train_y,train_mask,test_size=0.2)

    # tokenizer=BertTokenizer.from_pretrained('bert-base-chinese',do_lower_case=True)

    num_train_optimization_steps = None

    if args.do_train:
        train_examples = train_x
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    config = BertConfig('./nerout/bert_config.json')
    model = BertChineseNER(config, num_labels=num_labels)
    model.load_state_dict(torch.load('./nerout/pytorch_model.bin'))
    if args.fp16:
        model.half()
    model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        pass
        model = torch.nn.DataParallel(model,device_ids=[0,1,2])

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    eval_dataloader=None
    if args.do_eval:
        dev_all_input_ids = torch.tensor(dev_x, dtype=torch.long)
        dev_all_input_mask = torch.tensor(dev_mask, dtype=torch.long)
        dev_all_label_ids = torch.tensor(dev_y, dtype=torch.long)
        eval_data = TensorDataset(dev_all_input_ids, dev_all_input_mask, dev_all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    final_outs=[]
    # 验证
    if args.do_eval:
        print("***** Running evaluation *****")
        print("  Num examples = %d" % len(dev_x))
        print("  Batch size = %d" % args.eval_batch_size)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for input_ids, input_mask, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, attention_mask=input_mask, labels=label_ids)
                logits = model(input_ids, attention_mask=input_mask)

            _,out=torch.max(logits,dim=2)
            for i in range(out.size(0)):
                tmp=out[i,:].masked_select(input_mask[i,:].byte()).detach().cpu().numpy().tolist()
                final_outs.append(tmp)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            label_mask = input_mask.detach().cpu().numpy()
            tmp_eval_accuracy = accuracy(logits,label_ids,label_mask)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += label_mask.sum()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss / nb_tr_steps if args.do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': loss}
        for key in sorted(result.keys()):
            print(" {} = {}".format(key, str(result[key])))
    pass
    for i in range(len(final_outs)):
        test_str_list = tokenizer.convert_ids_to_tokens(dev_x[i])[1:sum(dev_mask[i]) - 1]
        print(''.join(test_str_list))
        res_tag = final_outs[i][1:sum(dev_mask[i]) - 1]
        res_tag = [inx2tag[res_tag[i]] for i in range(len(res_tag))]
        gold_tag = dev_y[i][1:sum(dev_mask[i]) - 1]
        gold_tag = [inx2tag[gold_tag[i]] for i in range(len(gold_tag))]
        getEnt(test_str_list, res_tag)
        print('Gold: ',end='')
        getEnt(test_str_list, gold_tag)






if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_dir",default=None,type=str,help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='bert-base-chinese', type=str, required=True,help="Bert pre-trained model selected in the list: bert-base-uncased, ""bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, ""bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",default='NER',type=str,help="The name of the task to train.")
    parser.add_argument("--output_dir",default=None,type=str,required=True,help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir",default="",type=str,help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",default=64,type=int,help="The maximum total input sequence length after WordPiece tokenization. \n""Sequences longer than this will be truncated, and sequences shorter \n""than this will be padded.")
    parser.add_argument("--do_train",action='store_true',help="Whether to run training.")
    parser.add_argument("--do_eval",action='store_true',help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",action='store_true',help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",default=256,type=int,help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",default=128,type=int,help="Total batch size for eval.")
    parser.add_argument("--learning_rate",default=5e-5,type=float,help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",default=3.0,type=float,help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",default=0.1,type=float,help="Proportion of training to perform linear learning rate warmup for. " "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",action='store_true',help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",type=int,default=-1,help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',type=int,default=42,help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',action='store_true',help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',type=float, default=0,help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n""0 (default value): dynamic loss scaling.\n""Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    main(args)