'''
Original code from ThuML's TSlib
https://github.com/thuml/Time-Series-Library
Edited by Arian Prabowo for DIEF.
'''
import argparse
import pprint
import torch
from exp.exp_classification_DIEF import Exp_Classification
import random
import numpy as np


def get_args(lsargs):
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='classification',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--seed', type=int, required=False, default=1, help='status')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')


    # data loader=
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=240, help='input sequence length')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--pos_weight', type=float, default=50, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # DIEF
    parser.add_argument('--exp_folder', type=str, default='./', help='experiment folders')
    parser.add_argument('--test_run', default=False, action="store_true", help="Load less data for code development")

    if lsargs is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(lsargs)
    return args


def run(lsargs=None):
    args = get_args(lsargs)

    if torch.cuda.is_available():
        args.use_gpu = True
        print('using cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        args.use_gpu = True
        print('using mps')
    else:
        args.use_gpu = False
        print('using cpu')

    print('Args in experiment:')
    pprint.pprint(vars(args))

    # SEED
    # fix_seed = 2021
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # EXPERIMENT
    exp = Exp_Classification(args)
    print('>>>>>>>start training : >>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train()
    print('>>>>>>>testing : <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    train_data, train_loader = exp._get_data(flag='TRAIN_ALL')
    total_loss, d_metrics = exp.vali(train_data, train_loader, exp._select_criterion())
    exp.de['trn_acc'] = np.array(d_metrics['accuracy']).mean().item()
    exp.de['trn_prc'] = np.array(d_metrics['precision']).mean().item()
    exp.de['trn_rec'] = np.array(d_metrics['recall']).mean().item()
    exp.de['trn_f1s'] = np.array(d_metrics['f1']).mean().item()
    exp.de['trn_mAP'] = np.array(d_metrics['AP']).mean().item()
    exp.de['trn_dMetrics'] = d_metrics
    exp.save_experiment_dict()

    exp.test()
    torch.cuda.empty_cache()

    # PRINT
    print(exp.de['exp_filepath'])
    print(exp.de['best_model_path'])
    print(exp.de['test_prediction_path'])
    return exp


if __name__ == '__main__':
    run()