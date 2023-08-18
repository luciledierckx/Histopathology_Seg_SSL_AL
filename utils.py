
import os, sys, time, random
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import pathlib
from query_strategies.util import get_unique_folder
from torch import nn
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    import warnings
    warnings.warn("tensorboard not installed: progress will not be stored")
    class SummaryWriter:
        "stub class to avoid errors with failed import of Tensorboard"
        def __init__(self, path: str) -> None:
            ...
        def add_scalar(self, label: str, value: float, idx: int) -> None:
            ...
        def flush(self) -> None:
            ...
        def close(self) -> None:
            ...


def print_log(string, log):
    print (string)
    with open(log, 'w+') as f:
        f.write(string)


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(
        time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)
        self._writer = None
        self.writer_path = "tensorboard"

    def set_writer_path(self, path: str):
        self.writer_path = path

    def get_writer(self) -> SummaryWriter:
        if self._writer is None:
            self._writer = SummaryWriter(str(get_unique_folder(pathlib.Path("save"), self.writer_path)))
        return self._writer

    def reset(self, total_epoch):
        assert total_epoch > 0
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2),
                                     dtype=np.float32)  # [epoch, train/val]
        self.epoch_losses = self.epoch_losses - 1

        self.epoch_dsc = np.zeros((self.total_epoch, 2),
                                       dtype=np.float32)  # [epoch, train/val]
        self.epoch_dsc = self.epoch_dsc

        self.epoch_mcc = -1*np.ones((self.total_epoch, 2),
                                       dtype=np.float32)  # [epoch, train/val]
        self.epoch_mcc = self.epoch_mcc

    def update(self, idx, train_loss, train_dsc, train_mcc, val_loss, val_dsc, val_mcc):
        assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(
            self.total_epoch, idx)
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_dsc[idx, 0] = train_dsc
        self.epoch_dsc[idx, 1] = val_dsc
        self.epoch_mcc[idx, 0] = train_mcc
        self.epoch_mcc[idx, 1] = val_mcc
        self.current_epoch = idx + 1
        # return self.max_accuracy(False) == val_acc
        self.get_writer().add_scalar("Loss/Train", train_loss, idx)
        self.get_writer().add_scalar("Loss/Val", val_loss, idx)
        self.get_writer().add_scalar("DiceScore/Train", train_dsc, idx)
        self.get_writer().add_scalar("DiceScore/Val", val_dsc, idx)
        self.get_writer().add_scalar("MCC/Train", train_mcc, idx)
        self.get_writer().add_scalar("MCC/Val", val_mcc, idx)
        self.get_writer().flush()

    def max_dsc(self, istrain):
        if self.current_epoch <= 0: return 0
        if istrain: return self.epoch_dsc[:self.current_epoch, 0].max()
        else: return self.epoch_dsc[:self.current_epoch, 1].max()
    def max_mcc(self, istrain):
        if self.current_epoch <= 0: return 0
        if istrain: return self.epoch_mcc[:self.current_epoch, 0].max()
        else: return self.epoch_mcc[:self.current_epoch, 1].max()

    def last_dsc(self, istrain):
        if self.current_epoch <= 0: return 0
        if istrain: return self.epoch_dsc[self.current_epoch-1, 0]
        else: return self.epoch_dsc[self.current_epoch-1, 1]
    def last_mcc(self, istrain):
        if self.current_epoch <= 0: return 0
        if istrain: return self.epoch_mcc[self.current_epoch-1, 0]
        else: return self.epoch_mcc[self.current_epoch-1, 1]

    def plot_curve(self, save_path, metric='dsc'):
        self.get_writer().close()
        title = 'the '+metric+'/loss curve of train/val'
        dpi = 80
        width, height = 1200, 800
        legend_fontsize = 10
        scale_distance = 48.8
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 1)
        interval_y = 0.05
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 1 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel(metric, fontsize=16)

        y_axis[:] = self.epoch_dsc[:, 0] if metric == 'dsc' else self.epoch_mcc[:, 0]
        plt.plot(x_axis,
                 y_axis,
                 color='g',
                 linestyle='-',
                 label='train-'+metric,
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_dsc[:, 1] if metric == 'dsc' else self.epoch_mcc[:,1]
        plt.plot(x_axis,
                 y_axis,
                 color='y',
                 linestyle='-',
                 label='val-'+metric,
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis,
                 y_axis,
                 color='g',
                 linestyle=':',
                 label='train-loss',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis,
                 y_axis,
                 color='y',
                 linestyle=':',
                 label='val-loss',
                 lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)


        if save_path is not None:
            fig.savefig(save_path+'_'+metric, dpi=dpi, bbox_inches='tight')
            print('---- save figure {} into {}'.format(title, save_path+'_'+metric))
        plt.close(fig)


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def time_file_str():
    ISOTIMEFORMAT = '%Y-%m-%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT,
                                       time.gmtime(time.time())))
    return string + '-{}'.format(random.randint(1, 10000))


def adjust_learning_rate(optimizer, epoch, gammas, schedule, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    "Add by YU"
    lr = args.lr
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu

def dice_loss(pred,target, reduction='mean'):
    numerator = 2 * torch.mul(pred,target)
    denominator = pred + target
    if reduction == "mean": return (1 - (numerator + 1) / (denominator + 1)).mean()
    elif reduction == "sum": return torch.mean((1 - (numerator + 1) / (denominator + 1)),(1,2)).sum()
    else: return torch.mean(1 - (numerator + 1) / (denominator + 1), (1,2))

def dice_score(pred,target, reduction='mean'):
    numerator = 2 * torch.mul(pred,target)
    denominator = pred + target
    if reduction == "mean": return ((numerator + 1) / (denominator + 1)).mean()
    elif reduction == "sum": return torch.mean(((numerator + 1) / (denominator + 1)),(1,2)).sum()
    else: return torch.mean((numerator + 1) / (denominator + 1), (1,2))

def plot_test_metrics(metric, dsc, mcc, perc_labeled, args, save_path=None):
    print('dsc', dsc) if metric == 'dsc' else print('mcc', mcc)
    title = 'the '+metric+' curve in function of labeled samples percentage'
    dpi = 80
    width, height = 1200, 800
    legend_fontsize = 10
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = perc_labeled  # percentage of labeled data
    y_axis = dsc if metric == 'dsc' else mcc

    plt.xlim(0, 1)
    plt.ylim(0, 1) if metric == 'dsc' else plt.ylim(-1,1)
    interval_y = 0.05
    interval_x = 0.05
    plt.xticks(np.arange(0, 1 + interval_x, interval_x))
    plt.yticks(np.arange(0, 1 + interval_y, interval_y)) if metric == 'dsc' else plt.yticks(np.arange(-1, 1 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('percentage of labeled samples', fontsize=16)
    plt.ylabel(metric, fontsize=16)

    y_axis[:] = dsc if metric == 'dsc' else mcc
    plt.plot(x_axis,
                y_axis,
                color='g',
                linestyle='-',
                marker='x',
                lw=2)

    if save_path is not None:
        filepath = save_path+'_final_test_'+metric+'_'+str(args.strategy)+'_'+str(args.query_strategy)+'_proRemoveGland'+str(args.probRemoveGland)+'_doFullySup'+str(args.doFullySupervized)+'_nepoch'+str(args.n_epoch)+'.png'
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print('---- save figure {} into {}'.format(title, filepath))
    plt.close(fig)

def save_test_metrics(dsc, mcc, perc_labeled, args, save_path):
    out = np.column_stack((perc_labeled,dsc, mcc))
    filepath = save_path+'_final_test_'+str(args.strategy)+'_'+str(args.query_strategy)+'_proRemoveGland'+str(args.probRemoveGland)+'_doFullySup'+str(args.doFullySupervized)+'_nepoch'+str(args.n_epoch)+'.csv'
    np.savetxt(filepath,out,delimiter=',', header="percentage labeled, DSC, MCC")

    
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model,epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if (self.counter >= self.patience) and (epoch>120):
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss