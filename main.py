### This project was based on the code structure of the Deep Active Learning repository from Cure Lab
### https://github.com/cure-lab/deep-active-learning
### The code was adapted to be applied to segmentation tasks and to use specific SSL and AL methods

import numpy as np
import random
import sys

import os
import argparse
from dataset import get_dataset, get_handler
from torchvision import transforms
import torch
import csv
import time
from prepare_data.transformations import Fliplr_image, Flipud_image, Rot90_image, Rot180_image
import query_strategies 
import models
from utils import print_log, plot_test_metrics, save_test_metrics

os.environ['CUBLAS_WORKSPACE_CONFIG']= ':16:8'
query_strategies_name = sorted(name for name in query_strategies.__dict__
                     if callable(query_strategies.__dict__[name]))
model_name = sorted(name for name in models.__dict__)

###############################################################################
parser = argparse.ArgumentParser()
# strategy
parser.add_argument('--strategy', help='acquisition algorithm', type=str, choices=query_strategies_name, 
                    default='rand')
parser.add_argument('--query_strategy', type=str, default=None,
                    help='(opt.) other strategy to query: must be compatible with strategy',
                    choices=query_strategies_name)
parser.add_argument('--nQuery',  type=float, default=5,
                    help='number of points to query in a batch (%)')
#parser.add_argument('--nStart', type=float, default=10,
#                    help='number of points to start (%)')
parser.add_argument('--probRemoveGland', type=float, default=0.4,
                     help='probability that a gland annotation will be removed an used as an unsupervized label initialy [0,1]')
parser.add_argument('--doFullySupervized', type=bool, default=False,
                     help='Set to True if you want to run the code as if the processed dataset was fully supervized (but still applying the gland annotation removal parameter)')
parser.add_argument('--nEnd',type=float, default=100,
                        help = 'total number of points to query (%)')
parser.add_argument('--nEmb',  type=int, default=256,
                        help='number of embedding dims (mlp)')
parser.add_argument('--seed', type=int, default=10,
                    help='the index of the repeated experiments', )
parser.add_argument('--device', type=str, default='0',
                    help='GPU device', )
parser.add_argument('--batch_size', type=int, default=4,
                    help='the batch size')
parser.add_argument("--save-image-freq", type=int, default=30,
                    help="save images after every N epochs. Skip saving with -1")
parser.add_argument("--allow_no_tensorboard", action="store_false", dest="tensorboard",
                    help="if False or unset, tensorboard must be installed for the script to run")
parser.set_defaults(tensorboard=True)

# model and data
parser.add_argument('--model', help='model - resnet, vgg, or mlp', type=str)
parser.add_argument('--dataset', help='dataset (non-openML)', type=str, default='')
parser.add_argument('--data_path', help='data path', type=str, default='./datasets')
parser.add_argument('--save_path', help='result save save_dir', default='./save')
parser.add_argument('--save_file', help='result save save_dir', default='result.csv')

# for gcn, designed for uncertainGCN and coreGCN
parser.add_argument("-n","--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")
parser.add_argument("-r","--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-l","--lambda_loss",type=float, default=1.2, 
                    help="Adjustment graph loss parameter between the labeled and unlabeled")
parser.add_argument("-s","--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")

# for ensemble based methods
parser.add_argument('--n_ensembles', type=int, default=1, 
                    help='number of ensemble')

# for proxy based selection
parser.add_argument('--proxy_model', type=str, default=None,
                    help='the architecture of the proxy model')

# training hyperparameters
parser.add_argument('--optimizer',
                    type=str,
                    default='SGD',
                    choices=['SGD', 'Adam', 'YF'])
parser.add_argument('--n_epoch', type=int, default=200,
                    help='number of training epochs in each iteration')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate. 0.01 for semi')
parser.add_argument('--gammas',
                    type=float,
                    nargs='+',
                    default=[0.1, 0.1],
                    help=
                    'LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--save_model', 
                    action='store_true',
                    default=False, help='save model every steps')
parser.add_argument('--load_ckpt', 
                    action='store_true',
                    help='load model from memory, True or False')
parser.add_argument('--add_imagenet', 
                    action='store_true',
                    help='load model from memory, True or False')

# automatically set
# parser.add_argument("--local_rank", type=int)

##########################################################################
args = parser.parse_args()

if args.tensorboard:
    # start the import to make sure it exists
    import tensorboard
    del tensorboard

# set the backend of the distributed parallel
# ngpus = torch.cuda.device_count()
# dist.init_process_group("nccl")

############################# For reproducibility #############################################
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
    torch.cuda.manual_seed(args.seed)
    # True ensures the algorithm selected by CUFA is deterministic
    # torch.backends.cudnn.deterministic = True
    # torch.set_deterministic(True)
    # False ensures CUDA select the same algorithm each time the application is run
    torch.backends.cudnn.benchmark = False
else:
    print('GPU not available')

############################# Specify the hyperparameters #######################################
 
args_pool = {'glas':
                {
                 'n_class':2,
                 'channels':3,
                 'input_size': 256,
                 'transform_tr': transforms.Compose([Fliplr_image(),Flipud_image(),Rot180_image(),Rot90_image()]),
                 'transform_te': None, 
                 'transform_val':None,      
                 'loader_tr_args':{'batch_size': args.batch_size, 'num_workers': 0},
                 'loader_te_args':{'batch_size': args.batch_size, 'num_workers': 0},
                 'loader_val_args':{'batch_size': args.batch_size, 'num_workers': 0}
                 }
        }

###############################################################################
###############################################################################

def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)
    log = os.path.join(args.save_path,
                        'log_seed_{}.txt'.format(args.seed))
    # print the args
    print(args.save_model)
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(str(state), log)
    print_log("Random Seed: {}".format(args.seed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    # load the dataset specific parameters
    dataset_args = args_pool[args.dataset]
    args.n_class = dataset_args['n_class']
    args.img_size = dataset_args['input_size']
    args.channels = dataset_args['channels']
    args.transform_tr = dataset_args['transform_tr']
    args.transform_te = dataset_args['transform_te']
    # args.transform_val = dataset_args['transform_val']
    args.loader_tr_args = dataset_args['loader_tr_args']
    args.loader_te_args = dataset_args['loader_te_args']
    args.loader_val_args = dataset_args['loader_val_args']
    #args.normalize = dataset_args['normalize']
    args.log = log 

    # Remove some gland annotation in function of probRemoveGland param
    command = 'python prepare_data/removeAnn_extractPatches.py --labels_dir datasets/glas/train_labels/ --imgs_dir datasets/glas/train_samples/ --output_dir datasets/glas/train_patches/ --probRemove '+ str(args.probRemoveGland) +' --patchSize '+str(args.img_size)
    os.system(command)

    if not os.path.exists("./datasets/glas/test_patches/"):
        command = "python prepare_data/removeAnn_extractPatches.py --labels_dir datasets/glas/test_labels/ --imgs_dir datasets/glas/test_samples/ --output_dir datasets/glas/test_patches/ --probRemove 0 --patchSize " +str(args.img_size)
        os.system(command)
    
    if not os.path.exists("./datasets/glas/valid_patches/"):
        command = "python prepare_data/removeAnn_extractPatches.py --labels_dir datasets/glas/valid_labels/ --imgs_dir datasets/glas/valid_samples/ --output_dir datasets/glas/valid_patches/ --probRemove 0 --patchSize " +str(args.img_size)
        os.system(command)

    # load dataset
    X_tr, Y_tr, X_val, Y_val, X_te, Y_te,idxs_lb = get_dataset(args.dataset, args.data_path, args.doFullySupervized)
    print(np.unique(idxs_lb, return_counts=True))
    if type(X_tr) is list:
        X_tr = np.array(X_tr)
        Y_tr = torch.tensor(np.array(Y_tr))
        X_val = np.array(X_val)
        Y_val = torch.tensor(np.array(Y_val))
        X_te = np.array(X_te)
        Y_te = torch.tensor(np.array(Y_te))

    if type(X_tr[0]) is not np.ndarray:
        X_tr = X_tr.numpy()
        X_te = X_te.numpy()
        
    args.dim = np.shape(X_tr)[1:]
    handler = get_handler(args.dataset)

    n_pool = len(Y_tr)
    n_test = len(Y_te)

    args.nStart = int(idxs_lb.mean() * 100)
    print("init percentage of labeled samples", args.nStart)
    args.nEnd =  args.nEnd if args.nEnd != -1 else 100
    args.nQuery = args.nQuery if args.nQuery != -1 else (args.nEnd - args.nStart)
    
    NUM_INIT_LB = idxs_lb.sum()
    print('num init lb', NUM_INIT_LB)
    
    NUM_QUERY = int(args.nQuery*n_pool/100) if args.nStart!= 100 else 0
    NUM_ROUND = int((int(args.nEnd*n_pool/100) - NUM_INIT_LB)/ NUM_QUERY) if args.nStart!= 100 else 0
    if NUM_QUERY != 0:
        if (int(args.nEnd*n_pool/100) - NUM_INIT_LB)% NUM_QUERY != 0:
            NUM_ROUND += 1
    
    print_log("[init={:02d}] [query={:02d}] [end={:02d}]".format(NUM_INIT_LB, NUM_QUERY, int(args.nEnd*n_pool/100)), log)

    # load specified network
    net = models.__dict__[args.model](n_class=args.n_class)
        

    # selection strategy
    strategy = query_strategies.__dict__[args.strategy](X_tr, Y_tr, X_val, Y_val, idxs_lb, net, handler, args)

    print_log('Strategy {} successfully loaded...'.format(args.strategy), log)
    query_strategy = None
    if args.query_strategy:
        query_strategy = query_strategies.__dict__[args.query_strategy](X_tr, Y_tr, X_val, Y_val, idxs_lb, net, handler, args)
        if hasattr(strategy, "set_query_strategy"):
            strategy.set_query_strategy(query_strategy)
            print_log(f"Query strategy {args.query_strategy} successfully loaded...", log)
        else:
            raise ValueError(f"unable to load {args.query_strategy=!r}")


    alpha = 2e-3
    # load pretrained model
    if args.load_ckpt:
        strategy.load_model()
        idxs_lb = strategy.idxs_lb
    else:
        strategy.train(alpha=alpha, n_epoch=args.n_epoch)
    test_dsc, test_mcc, _ = strategy.predict(X_te, Y_te)
    dsc = np.zeros(NUM_ROUND+1)
    dsc[0] = test_dsc
    mcc = -1*np.ones(NUM_ROUND+1)
    mcc[0] = test_mcc
    perc_labeled = np.zeros(NUM_ROUND+1)
    perc_labeled[0] = idxs_lb.mean()
    print_log('==>> Testing dsc {}'.format(dsc[0]) + ' testing mcc {}'.format(mcc[0]), log)

    out_file = os.path.join(args.save_path, args.save_file)
    for rd in range(1, NUM_ROUND+1):
        print('Round {}/{}'.format(rd, NUM_ROUND), flush=True)
        labeled = len(np.arange(n_pool)[idxs_lb])
        if NUM_QUERY > int(args.nEnd*n_pool/100) - labeled:
            NUM_QUERY = int(args.nEnd*n_pool/100) - labeled
            
        # query new samples to go in the labeled set
        ts = time.time()
        output = strategy.query(NUM_QUERY)
        q_idxs = output
        idxs_lb[q_idxs] = True
        te = time.time()
        tp = te - ts
       
        # update
        print(rd)
        strategy.update(idxs_lb)
        last_valid_dsc, last_valid_mcc = strategy.train(alpha=alpha, n_epoch=args.n_epoch,n_iter = rd)
        test_dsc, test_mcc, _ = strategy.predict(X_te, Y_te)
        t_iter = time.time() - ts
        
        # round accuracy
        dsc[rd] = test_dsc
        mcc[rd] = test_mcc
        perc_labeled[rd] = idxs_lb.mean()
        print_log(str(sum(idxs_lb)) + '\t' + 'testing dsc {}'.format(dsc[rd]) + ' testing mcc {}'.format(mcc[rd]), log)

        print_log("logging...", log)
        with open(out_file, 'a+') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([
                            args.strategy,
                            args.seed,
                            'budget',
                            args.nEnd,
                            'nStart', 
                            args.nStart,
                            'nQuery',
                            args.nQuery,
                            'labeled',
                            min(args.nStart + args.nQuery*rd, args.nEnd),
                            'dscCompare',
                            dsc[0],
                            dsc[rd],
                            dsc[rd] - dsc[0],
                            'mccCompare',
                            mcc[0],
                            mcc[rd],
                            mcc[rd] - mcc[0],
                            't_query',
                            tp,
                            't_iter',
                            t_iter
                            ])

    print_log('success!', log)

    save_test_metrics(dsc, mcc, perc_labeled, args, os.path.join(args.save_path, args.dataset))
    plot_test_metrics('dsc', dsc, mcc, perc_labeled, args, save_path=os.path.join(args.save_path, args.dataset))
    plot_test_metrics('mcc', dsc, mcc, perc_labeled, args, save_path=os.path.join(args.save_path, args.dataset))


if __name__ == '__main__':
    main()