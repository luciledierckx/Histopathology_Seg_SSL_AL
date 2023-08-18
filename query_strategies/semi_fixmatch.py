import warnings
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import os
import numpy as np
import time
from .strategy import Strategy
from utils import time_string, AverageMeter, RecorderMeter, convert_secs2time, adjust_learning_rate, dice_loss, dice_score
from torchmetrics.classification import MatthewsCorrCoef
import random
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
from copy import deepcopy
from prepare_data.transformations import TransformUDA
from utils import EarlyStopping
pseudo_label_threshold = 0.95


class fixmatch(Strategy):
    """
    Our omplementation of the paper: Unsupervised Data Augmentation for Consistency Training
    https://arxiv.org/pdf/1904.12848.pdf
    Google Research, Brain Team, 2 Carnegie Mellon University
    """

    def __init__(self, X, Y, X_val, Y_val, idxs_lb, net, handler, args):
        super(fixmatch, self).__init__(X, Y, X_val, Y_val, idxs_lb, net, handler, args)

    def query(self, n):
        """
        n: number of data to query
        return the index of the selected data
        """
        if self.query_strategy is not None:
            return self.query_strategy.query(self, n)
        else:
            warnings.warn("default query strategy used for fixmatch, is it intended?")

        ## Entropy-based query
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
        log_probs = torch.log(probs)
        U = (probs*log_probs).sum((1,2,3))
        return idxs_unlabeled[U.sort()[1][:n]]

    def _train(self, epoch, loader_tr_labeled, loader_tr_unlabeled, optimizer):
        self.clf.train()
        dscFinal = 0.
        mccFinal = 0.
        train_loss = 0.
        iter_unlabeled = iter(loader_tr_unlabeled) if loader_tr_unlabeled!= None else None
        nb_batch = 0
        for batch_idx, (x, y, idxs) in enumerate(loader_tr_labeled):
            nb_batch = batch_idx
            try:
                if iter_unlabeled!= None: (inputs_u, inputs_u2), _, _ = next(iter_unlabeled)
                else: 
                    inputs_u = torch.Tensor([])
                    inputs_u2 = torch.Tensor([])
            except StopIteration:
                iter_unlabeled = iter(loader_tr_unlabeled)
                (inputs_u, inputs_u2), _, _ = next(iter_unlabeled)
            input_all = torch.cat((x, inputs_u, inputs_u2)).to(self.device)
            y = y.to(self.device)
            output_all = self.clf(input_all)
            output_sup = output_all[:len(x)]
            output_sup_sm = torch.softmax(output_sup.float(),dim=1)[:,1]
            output_unsup = output_all[len(x):]
            output_u, output_u2 = torch.chunk(output_unsup, 2)
            output_u2_sm = torch.softmax(output_u2.float(),dim=1)[:,1]

            ## Segmentation loss
            loss = dice_loss(output_sup_sm, y.float()) + F.binary_cross_entropy(output_sup_sm, y.float(), reduction='mean') 
            
            ## Unsupervised loss
            if iter_unlabeled != None:
                pseudo_label = torch.softmax(output_u.detach(), dim=1)
                max_probs, max_idx = torch.max(pseudo_label, dim=1)
                sample_max_probs = torch.mean(max_probs, (1,2),keepdim=True)
                mask = sample_max_probs.ge(pseudo_label_threshold).float()
                masked_loss = mask * F.binary_cross_entropy(output_u2_sm, max_idx.float(), reduction='none')
                unsup_loss = masked_loss.mean()

                loss += unsup_loss.item()
            train_loss += loss.item()

            dscFinal += dice_score(torch.max(output_sup, 1)[1], y, reduction='sum').data.item()
            mccFinal += MatthewsCorrCoef("binary", num_classes=2).to(self.device)(torch.max(output_sup, 1)[1],y).data.item()
            
            optimizer.zero_grad()
            loss.backward()

            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()

            if batch_idx % 10 == 0:
                print("[Batch={:03d}] [Loss={:.2f}]".format(batch_idx, loss))
        return dscFinal / len(loader_tr_labeled.dataset.X), mccFinal / nb_batch, train_loss/nb_batch

    def train(self, alpha=0.1, n_epoch=10, batch_ratio=0.6, callback=True,n_iter=0):
        self.clf =  deepcopy(self.net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # self.clf = nn.parallel.DistributedDataParallel(self.clf,
        # find_unused_parameters=True,
        # )
        self.clf = nn.DataParallel(self.clf).to(self.device)
        parameters = self.clf.parameters()
        optimizer = optim.SGD(parameters, lr=self.args.lr, weight_decay=5e-4, momentum=self.args.momentum)
        checkpoint_path = f"save/tensorboard_{self.args.strategy}_{self.args.query_strategy}_proRemoveGland{self.args.probRemoveGland}_doFullySup{self.args.doFullySupervized}_nepoch{self.args.n_epoch}_{n_iter}_best_model.pkl"
        early_stopping = EarlyStopping(patience=30,path=checkpoint_path,delta=0.001,verbose=True)
        min_val_loss = 99999
        min_val_model_state_dict = None
        
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        epoch_time = AverageMeter()
        recorder = RecorderMeter(n_epoch)
        recorder.set_writer_path("tensorboard_" +str(self.args.strategy)+'_'+str(self.args.query_strategy)+'_proRemoveGland'+str(self.args.probRemoveGland)+'_doFullySup'+str(self.args.doFullySupervized)+'_nepoch'+str(self.args.n_epoch))
        epoch = 0
        train_dsc = 0.
        train_mcc = 0.

        if idxs_train.shape[0] != 0:
            transform = self.args.transform_tr
            train_data_labeled = self.handler(self.X[idxs_train],
                                              self.Y.numpy()[idxs_train],
                                              transform=transform)
            loader_tr_labeled = DataLoader(train_data_labeled,
                                           shuffle=True,
                                           pin_memory=True,
                                           # sampler = DistributedSampler(train_data),
                                           worker_init_fn=self.seed_worker,
                                           generator=self.g,
                                           **self.args.loader_tr_args)
        else: loader_tr_labeled = None
        if idxs_unlabeled.shape[0] != 0:
            train_data_unlabeled = self.handler(self.X[idxs_unlabeled],
                                                self.Y.numpy()[idxs_unlabeled],
                                                transform=TransformUDA(size=self.args.img_size,channels=self.args.channels),
                                                weak_strong=True)
            loader_tr_unlabeled = DataLoader(train_data_unlabeled,
                                             shuffle=True,
                                             pin_memory=True,
                                             # sampler = DistributedSampler(train_data),
                                             worker_init_fn=self.seed_worker,
                                             generator=self.g,
                                             **self.args.loader_tr_args)
        else: loader_tr_unlabeled = None
        for epoch in range(n_epoch):
            ts = time.time()
            current_learning_rate, _ = adjust_learning_rate(optimizer, epoch, self.args.gammas, self.args.schedule,
                                                            self.args)

            # Display simulation time
            need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (n_epoch - epoch))
            need_time = '[{} Need: {:02d}:{:02d}:{:02d}]'.format(self.args.strategy, need_hour, need_mins,
                                                                    need_secs)

            # train one epoch
            train_dsc, train_mcc, train_los = self._train(epoch, loader_tr_labeled, loader_tr_unlabeled, optimizer)
            val_dsc, val_mcc, val_loss = self.predict(self.X_val, self.Y_val)
            
            # measure elapsed time
            epoch_time.update(time.time() - ts)
            recorder.update(epoch, train_los, train_dsc, train_mcc, val_loss, val_dsc, val_mcc)
            print('==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}]'.format(time_string(), epoch, n_epoch,
                                                                                need_time, current_learning_rate
                                                                                ) \
                    + ' [Current Train DSC={:.4f}]'.format(recorder.last_dsc(True))
                    + ' [Current Train MCC={:.4f}]'.format(recorder.last_mcc(True))
                    + '\n')            
            if callback and (min_val_loss==99999 or min_val_loss>val_loss): 
                print("callback, new best epoch is epoch", epoch)
                min_val_loss=val_loss
                min_val_model_state_dict = self.clf.state_dict()
            
            early_stopping(val_loss, self.clf,epoch)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.can_save_images(epoch):
                self.save_some_images()

        if self.args.save_model:
            self.save_model()
        if callback: self.clf.load_state_dict(min_val_model_state_dict)
        recorder.plot_curve(os.path.join(self.args.save_path, self.args.dataset),metric='dsc')
        recorder.plot_curve(os.path.join(self.args.save_path, self.args.dataset),metric='mcc')
        self.clf = self.clf.module

        last_val_dsc = recorder.last_dsc(istrain=False)
        last_val_mcc = recorder.last_mcc(istrain=False)
        return last_val_dsc, last_val_mcc
