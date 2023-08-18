from joblib.externals.cloudpickle.cloudpickle import instance
import numpy as np
import random
from sklearn import preprocessing
from torch import nn
import sys, os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
from utils import time_string, AverageMeter, RecorderMeter, convert_secs2time, adjust_learning_rate, dice_score, dice_loss
import time
from torchvision.utils import save_image
from tqdm import tqdm
from .util import get_unique_folder
from sklearn.metrics import pairwise_distances
from torchmetrics.classification import MatthewsCorrCoef
import pathlib

class Strategy:
    def __init__(self, X, Y, X_val, Y_val, idxs_lb, net, handler, args):
        self.X = X  # vector
        self.Y = Y
        self.X_val = X_val
        self.Y_val = Y_val

        self.idxs_lb = idxs_lb # bool type
        self.handler = handler
        self.args = args
        
        self.n_pool = len(Y)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.net = net.to(self.device)
        self.clf = deepcopy(net.to(self.device))

        # for reproducibility
        self.g = torch.Generator()
        self.g.manual_seed(0)

        self.query_strategy: "QueryStrategy | None" = None
        self.save_imgs_freq = max(int(args.save_image_freq), -1)
        assert self.save_imgs_freq != 0, "0 is invalid for save_imgs_freq"

    def seed_worker(self, worker_id):
        """
        To preserve reproducibility when num_workers > 1
        """
        # https://pytorch.org/docs/stable/notes/randomness.html
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def set_query_strategy(self, query_strategy: "QueryStrategy | None"):
        self.query_strategy = query_strategy

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    # Train one epoch
    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()

        dscFinal = 0.
        mccFinal = 0.
        train_loss = 0.
        nb_batch = 0
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            nb_batch += 1
            x, y = x.to(self.device), y.to(self.device) 
            nan_mask = torch.isnan(x)
            if nan_mask.any():
                raise RuntimeError(f"Found NAN in input indices: ", nan_mask.nonzero())

            # exit()
            optimizer.zero_grad()

            out, e1 = self.clf(x)
            pred = out.max(1)[1]
            nan_mask_out = torch.isnan(y)
            if nan_mask_out.any():
                raise RuntimeError(f"Found NAN in output indices: ", nan_mask.nonzero())
                
            loss = F.cross_entropy(out, y)

            train_loss += loss.item()
            dscFinal += dice_score(pred, y, reduction='sum').data.item()
            mccFinal += MatthewsCorrCoef("binary", num_classes=2).to(self.device)(pred,y).data.item()
            loss.backward()
            
            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
            
            if batch_idx % 10 == 0:
                print ("[Batch={:03d}] [Loss={:.2f}]".format(batch_idx, loss))

        return dscFinal / len(loader_tr.dataset.X), mccFinal/nb_batch, train_loss

    # Train all epochs
    def train(self, alpha=0.1, n_epoch=10):
        self.clf =  deepcopy(self.net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # self.clf = nn.parallel.DistributedDataParallel(self.clf,
                                                        # find_unused_parameters=True,
                                                        # )
        self.clf = nn.DataParallel(self.clf).to(self.device)
        parameters = self.clf.parameters()
        optimizer = optim.SGD(parameters, lr = self.args.lr, weight_decay=5e-4, momentum=self.args.momentum)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        

        epoch_time = AverageMeter()
        recorder = RecorderMeter(n_epoch)
        epoch = 0 
        train_dsc = 0.
        best_test_dsc = 0.
        train_mcc = 0.
        best_test_mcc = 0.
        if idxs_train.shape[0] != 0:
            transform = self.args.transform_tr

            train_data = self.handler(self.X[idxs_train], 
                                torch.Tensor(self.Y[idxs_train]).long() if type(self.Y) is np.ndarray else  torch.Tensor(self.Y.numpy()[idxs_train]).long(), 
                                    transform=transform)

            loader_tr = DataLoader(train_data, 
                                    shuffle=True,
                                    pin_memory=True,
                                    # sampler = DistributedSampler(train_data),
                                    worker_init_fn=self.seed_worker,
                                    generator=self.g,
                                    **self.args.loader_tr_args)
            for epoch in range(n_epoch):
                ts = time.time()
                current_learning_rate, _ = adjust_learning_rate(optimizer, epoch, self.args.gammas, self.args.schedule, self.args)
                
                # Display simulation time
                need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (n_epoch - epoch))
                need_time = '[{} Need: {:02d}:{:02d}:{:02d}]'.format(self.args.strategy, need_hour, need_mins, need_secs)
                
                # train one epoch
                train_dsc, train_mcc, train_los = self._train(epoch, loader_tr, optimizer)
                test_dsc, test_mcc = self.predict(self.X_val, self.Y_val)
                # measure elapsed time
                epoch_time.update(time.time() - ts)
                print('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}]'.format(time_string(), epoch, n_epoch,
                                                                                   need_time, current_learning_rate
                                                                                   ) \
                + ' [Best : Test Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               1. - recorder.max_accuracy(False)))
                recorder.update(epoch, train_los, train_dsc,train_mcc, 0, test_dsc, test_mcc)

                if self.args.save_model and test_dsc > best_test_dsc:
                    best_test_dsc = test_dsc
                    self.save_model()
                if self.args.save_model and test_mcc > best_test_mcc:
                    best_test_mcc = test_mcc
            recorder.plot_curve(os.path.join(self.args.save_path, self.args.dataset))
            self.clf = self.clf.module

        best_test_dsc = recorder.max_dsc(istrain=False)
        best_test_mcc = recorder.max_mcc(istrain=False)
        return best_test_dsc, best_test_mcc                


    def predict(self, X, Y):
        transform=self.args.transform_te 
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True, 
                        shuffle=False, **self.args.loader_te_args)
        
        self.clf.eval()

        correct = 0.
        dsc = 0.
        mcc = 0.
        loss = 0.
        with torch.no_grad():
            nb_batch = 0
            for x, y, idxs in loader_te:
                nb_batch += 1
                x, y = x.to(self.device), y.to(self.device) 
                out = torch.softmax(self.clf(x), dim=1)
                pred = out.max(1)[1]   
                loss += dice_loss(out[:,1], y.float()) + F.binary_cross_entropy(out[:,1], y.float(), reduction='mean')
                dsc += dice_score(pred, y, reduction='sum').data.item()
                mcc += MatthewsCorrCoef("binary").to(self.device)(pred,y).data.item()             
                #correct +=  (y == pred).sum().item() 

            #test_acc = correct / (len(Y)*Y.shape[1]*Y.shape[2])
            test_dsc = dsc / len(Y)
            test_mcc = mcc / nb_batch
            test_loss = loss/ nb_batch
   
        return test_dsc, test_mcc, test_loss.item()

    def get_prediction(self, X, Y):
        transform=self.args.transform_te 
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True, 
                        shuffle=False, **self.args.loader_te_args)

        P = torch.zeros(len(X)).long().to(self.device)

        self.clf.eval()


        correct = 0
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                out, e1 = self.clf(x)
                pred = out.max(1)[1]     
                P[idxs] = pred           
                correct +=  (y == pred).sum().item() 
   
        return P

    def predict_prob(self, X, Y):
        transform = self.args.transform_te 
        loader_te = DataLoader(self.handler(X, Y, 
                        transform=transform), shuffle=False, pin_memory=True, **self.args.loader_te_args)

        self.clf.eval()
        probs = torch.zeros([len(Y), self.args.n_class, Y.shape[1], Y.shape[2]])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu().data
        return probs


    def avg_fn(self, X: torch.Tensor, Y: torch.Tensor, reduction) -> torch.Tensor:
        """reduction for each image

        Args:
            X (torch.Tensor): float[N, 3, H, W]
            Y (torch.Tensor): int[N, H, W]
            reduction ((prob: float[N, C, H, W], y: int[N, H, W]) -> float[N])
        """
        transform = self.args.transform_te 
        loader_te = DataLoader(self.handler(X, Y, 
                        transform=transform), shuffle=False, pin_memory=True, **self.args.loader_te_args)

        self.clf.eval()
        reduced_lst = torch.zeros([len(Y)])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                prob = F.softmax(out, dim=1)
                
                reduced = reduction(prob, y)
                reduced_lst[idxs] = reduced.cpu().data

        return reduced_lst

    def predict_prob_dropout(self, X, Y, n_drop):
        transform = self.args.transform_te 
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True,
                            shuffle=False, **self.args.loader_te_args)

        self.clf.train()

        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device) 
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu().data
        probs /= n_drop
        
        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        transform = self.args.transform_te
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True,
                            shuffle=False, **self.args.loader_te_args)

        self.clf.train()

        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device) 
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu().data
            return probs

    def get_embedding(self, X, Y):
        """ get last layer embedding from current model"""
        transform = self.args.transform_te
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True,
                            shuffle=False, **self.args.loader_te_args)

        self.clf.eval()
        
        embedding = torch.zeros([len(Y), 
                self.clf.module.get_embedding_dim() if isinstance(self.clf, nn.DataParallel) 
                else self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                out, e1 = self.clf(x)
                embedding[idxs] = e1.data.cpu().float()
        
        return embedding


    def get_grad_embedding(self, X, Y):
        """ gradient embedding (assumes cross-entropy loss) of the last layer"""
        transform = self.args.transform_te 

        model = self.clf
        if isinstance(model, nn.DataParallel):
            model = model.module
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True,
                            shuffle=False, **self.args.loader_te_args)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                cout, out = self.clf(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)
    
    def save_model(self):
        # save model and selected index
        save_path = os.path.join(self.args.save_path,self.args.dataset+'_checkpoint')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        labeled = len(np.arange(self.n_pool)[self.idxs_lb])
        labeled_percentage = '%.1f'%float(100*labeled/len(self.X))
        torch.save(self.clf, os.path.join(save_path, self.args.strategy+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args.seed)+'.pkl'))
        print('save to ',os.path.join(save_path, self.args.strategy+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args.seed)+'.pkl'))
        path = os.path.join(save_path, self.args.strategy+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args.seed)+'.npy')
        np.save(path,self.idxs_lb)

    def load_model(self):
        labeled = len(np.arange(self.n_pool)[self.idxs_lb])
        labeled_percentage = '%.1f'%float(100*labeled/len(self.X))
        save_path = os.path.join(self.args.save_path,self.args.dataset+'_checkpoint')
        self.clf = torch.load(os.path.join(save_path, self.args.strategy+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args.seed)+'.pkl'))
        self.idxs_lb = np.load(os.path.join(save_path, self.args.strategy+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args.seed)+'.npy'))

    def can_save_images(self, epoch: int):
        if self.save_imgs_freq < 0:
            return False
        return (epoch + 1) % self.save_imgs_freq == 0

    def save_some_images(self, n: int = 10):
        "save approx. n images in the test set with predictions"

        transform=self.args.transform_te 
        loader_te = DataLoader(self.handler(self.X_val, self.Y_val, transform=transform), pin_memory=True, 
                        shuffle=False, **self.args.loader_te_args)

        base = get_unique_folder(pathlib.Path("images") / "save", "masks")

        dataset = loader_te.dataset

        self.clf.eval()
        with torch.no_grad():
            for idx in range(0, len(dataset), len(dataset) // n):
                x, y, _ = dataset[idx]
                y = y.float()

                pred = self.clf(torch.unsqueeze(x, 0).to(self.device))
                _, pred_cls = torch.max(pred[0], dim=0)
                pred_cls = pred_cls.float().cpu()

                save_image(x, base / f"ds-{idx}-x.png")
                save_image(y, base / f"ds-{idx}-y.png")
                save_image(pred_cls, base / f"ds-{idx}-pred_cls.png")


class QueryStrategy:
    def __init__(self, *_, **__) -> None:
        pass
    def query(self, strategy: Strategy, n: int) -> torch.Tensor:
        pass
