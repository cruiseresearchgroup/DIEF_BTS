'''
Original code from ThuML's TSlib
https://github.com/thuml/Time-Series-Library
Edited by Arian Prabowo for DIEF.
'''
import os
import copy
import json
from tqdm.auto import tqdm
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import time
import warnings
import numpy as np
import diefComp1Utils as util

warnings.filterwarnings('ignore')

# This is the ratio of negative to positive samples in the training set.
# This is used to calculate the pos_weight for the BCEWithLogitsLoss.
# https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
# Arian's personal note: e04_9_imbalance.ipynb
# max value is 10616.0
a_ratio = [8.52197309e+00, 2.42785714e+01, 1.09593750e+02, 2.15673469e+02,
           4.60608696e+02, 2.42785714e+01, 4.16385542e+01, 1.23905882e+02,
           0.00000000e+00, 1.76850000e+03, 1.76850000e+03, 1.57990506e+01,
           2.31845103e+01, 1.31712500e+02, 0.00000000e+00, 1.19951040e+01,
           3.78178571e+02, 6.88486842e+01, 6.00172414e+01, 2.39225352e+01,
           2.39225352e+01, 2.20303688e+01, 3.78178571e+02, 0.00000000e+00,
           3.41556291e+01, 1.09593750e+02, 1.59863636e+02, 2.07176471e+02,
           1.04118812e+02, 2.07176471e+02, 1.62338462e+02, 4.81590909e+02,
           1.25392857e+02, 1.84807339e+01, 3.31382637e+01, 5.29850000e+02,
           2.39225352e+01, 2.65325000e+03, 6.18224852e+01, 6.37378049e+01,
           2.80081967e+01, 0.00000000e+00, 1.06160000e+04, 3.34707792e+01,
           2.24893617e+02, 6.62562500e+02, 3.71906475e+01, 1.70561224e+01,
           1.32612500e+03, 1.51571429e+03, 2.34933333e+02, 7.22206897e+01,
           2.15673469e+02, 0.00000000e+00, 8.74750000e+01, 2.65325000e+03,
           4.23680000e+02, 4.16385542e+01, 1.23905882e+02, 4.28719008e+01,
           1.30074074e+02, 1.31712500e+02, 1.59863636e+02, 2.62930591e+01,
           3.63838028e+01, 6.10167224e+00, 3.32483871e+01, 5.96685714e+01,
           1.46458333e+02, 1.97769080e+01, 1.78949153e+02, 8.32619048e+01,
           4.60608696e+02, 4.81590909e+02, 1.01086538e+02, 5.18208955e+01,
           1.89193246e+01, 2.12578616e+01, 2.72367021e+01, 3.21781250e+01,
           7.22206897e+01, 6.00172414e+01, 3.67829181e+01, 1.23905882e+02,
           5.93238636e+01, 4.16385542e+01, 8.82184874e+01, 6.62562500e+02,
           2.39225352e+01, 2.65325000e+03, 6.18224852e+01, 6.37378049e+01,
           1.48535211e+02, 3.39243421e+01, 0.00000000e+00, 2.07176471e+02,
           1.07336735e+02, 1.70241935e+02, 3.71906475e+01, 0.00000000e+00,
           3.48682432e+01, 0.00000000e+00, 1.02077670e+02, 1.76850000e+03,
           4.81590909e+02, 0.00000000e+00, 6.00172414e+01, 1.78244681e+01,
           2.80081967e+01, 0.00000000e+00, 0.00000000e+00, 6.00172414e+01,
           9.73055556e+01, 2.98633721e+01, 3.31382637e+01, 6.00172414e+01,
           3.53800000e+03, 1.25392857e+02, 3.99922780e+01, 3.80330882e+01,
           0.00000000e+00, 3.53800000e+03, 6.62562500e+02, 3.53800000e+03,
           3.99922780e+01, 3.53800000e+03, 1.25392857e+02, 3.99922780e+01,
           3.53595890e+01, 4.60608696e+02, 6.62562500e+02, 3.99922780e+01,
           5.18208955e+01, 2.98633721e+01, 1.09158249e+01, 2.65325000e+03,
           4.81590909e+02, 3.51122449e+01, 2.65325000e+03, 3.02342857e+02,
           3.28121019e+01, 2.51785714e+02, 1.36883117e+02, 3.28121019e+01,
           3.28121019e+01, 2.62930591e+01, 6.00172414e+01, 1.09158249e+01,
           3.78178571e+02, 4.81590909e+02, 4.60608696e+02, 3.78178571e+02,
           9.37946429e+01, 4.53624454e+01, 7.94318182e+01, 4.53624454e+01,
           3.33592233e+01, 5.83128492e+01, 8.82184874e+01, 4.07346154e+02,
           1.45446559e+01, 6.20284939e+00, 0.00000000e+00, 1.06160000e+04,
           5.29850000e+02, 1.47756315e+01, 1.06160000e+04, 3.51122449e+01,
           5.88075178e+00, 2.99533528e+01, 4.95571429e+01, 1.62338462e+02,
           1.08453608e+02, 3.74673913e+01, 4.41375000e+02, 8.42895204e+00,
           1.02077670e+02, 7.06800000e+02, 2.65325000e+03, 0.00000000e+00,
           4.81590909e+02, 4.81590909e+02, 4.83813953e+01, 1.25392857e+02,
           8.00458015e+01, 0.00000000e+00, 2.93342857e+01, 3.70537634e+01,
           2.62930591e+01, 9.69758813e-01, 8.29684764e+00, 3.36960784e+01,
           3.36960784e+01, 4.60608696e+02, 2.78394737e+02, 3.70537634e+01,
           4.07346154e+02, 5.73351648e+01, 5.57789474e+02, 9.46486486e+01,
           6.62562500e+02, 0.00000000e+00, 4.20441176e+00, 4.23680000e+02,
           9.46486486e+01, 6.62562500e+02, 6.62562500e+02, 3.67829181e+01,
           1.14402174e+02, 1.25392857e+02, 2.65325000e+03, 0.00000000e+00,
           1.46458333e+02, 2.01073559e+01, 2.01073559e+01, 9.64181818e+02,
           1.15670330e+02, 1.97363281e+01, 1.28603133e+01, 1.64049180e+01,
           1.48535211e+02, 0.00000000e+00, 3.41483871e+02, 1.89193246e+01,
           8.32619048e+01, 1.84807339e+01, 1.09158249e+01, 1.98176471e+01,
           2.34068966e+01, 3.51122449e+01, 3.11264706e+02, 1.92036364e+02,
           6.61962025e+01, 1.25392857e+02, 4.07346154e+02, 2.78394737e+02,
           2.51785714e+02, 1.11946809e+02, 1.76850000e+03, 2.16375267e+01]
a_ratio = torch.tensor(a_ratio)

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.de = vars(args) # de: dictionary of experiment
        # Path and files
        self.de['ProcessID'] = os.getpid()
        self.de['ScriptTime'] = int(time.time())
        self.de['exp_fileheader'] = 'expDIEF'+str(self.de['ScriptTime'])+'_'+str(self.de['ProcessID'])
        self.de['exp_filepath'] = os.path.join(args.exp_folder, self.de['exp_fileheader']+'.json')
        if not os.path.exists(args.exp_folder):
            os.makedirs(args.exp_folder)
        pathCP = os.path.join(args.exp_folder, 'checkpoints')
        if not os.path.exists(pathCP):
            os.makedirs(pathCP)
        self.de['best_model_path'] = os.path.join(pathCP, self.de['exp_fileheader']+'.pth')
        pathH = os.path.join(args.exp_folder, 'test_H')
        if not os.path.exists(pathH):
            os.makedirs(pathH)
        self.de['test_prediction_path'] = os.path.join(pathH, self.de['exp_fileheader']+'.csv')
        # Time and memory complexity analysis
        self.de['train_start_time'] = -1
        self.de['train_end_time'] = -1
        self.de['train_duration'] = -1
        self.de['test_start_time'] = -1
        self.de['test_end_time'] = -1
        self.de['test_duration'] = -1
        self.de['GPU_allocated_memory'] = -1
        # Training
        self.de['list_iter_loss_trn'] = []
        self.de['list_epch_loss_trn'] = []
        self.de['list_epch_loss_val'] = []
        self.de['list_epch_mtrc_val'] = []
        self.de['list_epch_time'] = []
        # Train results
        self.de['trn_acc'] = -1
        self.de['trn_prc'] = -1
        self.de['trn_rec'] = -1
        self.de['trn_f1s'] = -1
        self.de['trn_mAP'] = -1
        self.de['trn_dMetrics'] = {}
        # Test results
        self.de['tstLB_acc'] = -1
        self.de['tstLB_prc'] = -1
        self.de['tstLB_rec'] = -1
        self.de['tstLB_f1s'] = -1
        self.de['tstLB_mAP'] = -1
        self.de['tstLB_dMetrics'] = {}
        self.de['tstSc_acc'] = -1
        self.de['tstSc_prc'] = -1
        self.de['tstSc_rec'] = -1
        self.de['tstSc_f1s'] = -1
        self.de['tstSc_mAP'] = -1
        self.de['tstSc_dMetrics'] = {}
        self.de['tstCm_acc'] = -1
        self.de['tstCm_prc'] = -1
        self.de['tstCm_rec'] = -1
        self.de['tstCm_f1s'] = -1
        self.de['tstCm_mAP'] = -1
        self.de['tstCm_dMetrics'] = {}
        self.save_experiment_dict()
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        self.args.num_class = 240 # DIEF hardcode
        self.args.device = self.device
        model = self.model_dict[self.args.model].Model(self.args).float()
        print(model)
        self.de['model_str_summary'] = repr(model)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optim, 'min')
        return model_optim

    def _select_criterion(self):
        # self.args.pos_weight is a low pass filter.
        # self.args.pos_weight is the maximum amount.
        a_pos_weight = torch.where(a_ratio<self.args.pos_weight,a_ratio, self.args.pos_weight).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight = a_pos_weight)
        return criterion

    def predict(self, dataset, loader):
        # IMPORTANT, NO SIGMOID YET!!!
        # note: trues is NOT boolean. It is -1, 0, 1
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in tqdm(enumerate(loader), desc='Predict'):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                outputs = self.model(batch_x, padding_mask, None, None)
                preds.append(outputs.detach().cpu())
                trues.append(label)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0) if dataset.have_Y else None
        # probs = torch.nn.functional.sigmoid(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        return preds, trues

    def predict_and_save(self, dataset, dataloader):
        '''
        Need to decide if we are saving the boolean predictions or the probabilities.
        File size is a concern.
        check evernote: 2024 05 16 log: Should participants submit bool or float?
        '''
        raise NotImplementedError()
        # preds, trues = self.predict(dataset, dataloader)
        # probs = torch.nn.functional.sigmoid(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        # util.save_lilrows()
        # self.save_experiment_dict()
        return

    def vali(self, vali_data, vali_loader, criterion):
        preds, trues = self.predict(vali_data, vali_loader)
        # note: preds is before sigmoied. The sigmoid is in the criterion.
        # note: trues is NOT boolean. It is -1, 0, 1
        total_loss = criterion(preds.to(self.device), (trues>=0).float().to(self.device)).cpu()
        # note: preds is before sigmoied. The probs is after.
        probs = torch.nn.functional.sigmoid(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        d_metrics = util.allMetrics(trues.cpu().numpy(), probs.cpu().numpy()) # DIEF metrics
        return total_loss, d_metrics

    def train(self):
        # Load data
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VALI')
        train_steps = len(train_loader)
        # Set up experiments
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        # Create containers to log progress
        self.de['list_iter_loss_trn'] = []
        self.de['list_epch_loss_trn'] = []
        self.de['list_epch_loss_val'] = []
        self.de['list_epch_mtrc_val'] = []
        self.de['list_epch_time'] = []
        # Time performance
        time_now = time.time()
        self.de['train_start_time'] = time.time()
        # Main training loop
        for epoch in tqdm(range(self.args.train_epochs), desc='Train'):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, label, padding_mask) in enumerate(tqdm(train_loader, desc='Epoch:' + str(epoch))):
                iter_count += 1
                model_optim.zero_grad()
                # prepare a batch of data
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                # forward
                outputs = self.model(batch_x, padding_mask, None, None)
                # note: outputs is before sigmoied. The sigmoid is in the criterion.
                # note: label is NOT boolean. It is -1, 0, 1
                loss = criterion(outputs, (label>=0).float()) # DIEF: BCEWithLogitsLoss
                # backward
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
                # log and print per iter results
                # print('end of iter: current_allocated_memory', torch.mps.current_allocated_memory())
                # print('end of iter: driver_allocated_memory', torch.mps.driver_allocated_memory())
                train_loss.append(loss.item())
                self.de['list_iter_loss_trn'].append(loss.item())
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
            epoch_duration = time.time() - epoch_time
            train_loss = np.average(train_loss)
            # Validate
            vali_loss, d_metrics = self.vali(vali_data, vali_loader, criterion)
            self.scheduler.step(vali_loss)
            # Early stopping
            early_stopping(vali_loss, self.model, self.de['best_model_path'])
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            # Log per epoch results
            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_duration))
            self.de['list_epch_time'].append(epoch_duration)
            self.de['list_epch_loss_trn'].append(train_loss.item())
            self.de['list_epch_loss_val'].append(vali_loss.item())
            self.de['list_epch_mtrc_val'].append(d_metrics)
            self.save_experiment_dict()
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss))
        # Log training results
        self.de['train_end_time'] = time.time()
        self.de['train_duration'] = self.de['train_end_time'] - self.de['train_start_time']
        if torch.cuda.is_available():
            self.de['GPU_allocated_memory'] = max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.de['GPU_allocated_memory'] = max(torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory())
        print('end of train: GPU_allocated_memory', self.de['GPU_allocated_memory'])
        self.save_experiment_dict()
        return self.model

    def test(self):
        test_data, test_loader = self._get_data(flag='TEST')
        if not test_data.have_Y:
            print('No test data')
            return None, None, None
        print('loading model')
        self.model.load_state_dict(torch.load(self.de['best_model_path']))
        preds = []
        self.model.eval()
        self.de['test_start_time'] = time.time()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                outputs = self.model(batch_x, padding_mask, None, None)
                preds.append(outputs.detach())
        self.de['test_end_time'] = time.time()
        self.de['test_duration'] = self.de['test_end_time'] - self.de['test_start_time']
        preds = torch.cat(preds, 0)
        print('test shape:', preds.shape, test_data.labels_df.shape)
        # note: preds is before sigmoied. probs is after.
        probs = torch.nn.functional.sigmoid(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        # DIEF metrics
        d_mLB = util.parition_wrapper_for_metrics(test_data.labels_df, probs.cpu().numpy(), 'leaderboard')
        self.de['tstLB_acc'] = np.array(d_mLB['accuracy']).mean().item()
        self.de['tstLB_prc'] = np.array(d_mLB['precision']).mean().item()
        self.de['tstLB_rec'] = np.array(d_mLB['recall']).mean().item()
        self.de['tstLB_f1s'] = np.array(d_mLB['f1']).mean().item()
        self.de['tstLB_mAP'] = np.array(d_mLB['AP']).mean().item()
        self.de['tstLB_dMetrics'] = d_mLB
        d_mSc = util.parition_wrapper_for_metrics(test_data.labels_df, probs.cpu().numpy(), 'secret')
        self.de['tstSc_acc'] = np.array(d_mSc['accuracy']).mean().item()
        self.de['tstSc_prc'] = np.array(d_mSc['precision']).mean().item()
        self.de['tstSc_rec'] = np.array(d_mSc['recall']).mean().item()
        self.de['tstSc_f1s'] = np.array(d_mSc['f1']).mean().item()
        self.de['tstSc_mAP'] = np.array(d_mSc['AP']).mean().item()
        self.de['tstSc_dMetrics'] = d_mSc
        d_mCm = util.parition_wrapper_for_metrics(test_data.labels_df, probs.cpu().numpy(), 'combined')
        self.de['tstCm_acc'] = np.array(d_mCm['accuracy']).mean().item()
        self.de['tstCm_prc'] = np.array(d_mCm['precision']).mean().item()
        self.de['tstCm_rec'] = np.array(d_mCm['recall']).mean().item()
        self.de['tstCm_f1s'] = np.array(d_mCm['f1']).mean().item()
        self.de['tstCm_mAP'] = np.array(d_mCm['AP']).mean().item()
        self.de['tstCm_dMetrics'] = d_mCm
        self.save_experiment_dict()
        return d_mLB, d_mSc, d_mCm
    
    def save_experiment_dict(self):
        dict_no_device = copy.deepcopy(self.de)
        if 'device' in self.de.keys():
            del dict_no_device['device']
        with open(self.de['exp_filepath'], 'w') as f:
            json.dump(dict_no_device, f, indent=4)
        return
