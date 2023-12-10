#%%
import dgl
import ipdb
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

from utils import *
from models import *
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch_geometric.utils import dropout_adj, convert
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee
import torch.nn as nn
from torch_sparse import SparseTensor


"""
save data
"""
class Data:
    def __init__(self, edge_index, features, labels, idx_train, idx_val, idx_test, sens, sens_idx):
        self.edge_index = edge_index
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.sens = sens
        self.sens_idx = sens_idx


def run(args):
    """
    Load data
    """
    # Load bail dataset
    if args.dataset == 'bail':
        sens_attr = "WHITE"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "RECID"
        label_number = 100
        path_bail = "./datasets/bail"
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(args.dataset, sens_attr,
                                                                              predict_attr, path=path_bail,
                                                                              label_number=label_number,
                                                                              )
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features
    # load pokec dataset
    elif args.dataset == 'pokec_z':
        dataset = 'region_job'
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        label_number = 4000
        sens_number = 200
        sens_idx = 3
        seed = 20
        path = "./datasets/pokec/"
        test_idx = False
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                               sens_attr,
                                                                                               predict_attr,
                                                                                               path=path,
                                                                                               label_number=label_number,
                                                                                               sens_number=sens_number,
                                                                                               seed=seed,
                                                                                               test_idx=test_idx)
        labels[labels > 1] = 1
    elif args.dataset == 'pokec_n':
        dataset = 'region_job_2'
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        label_number = 3500
        sens_number = 200
        sens_idx = 3
        seed = 20
        path = "./datasets/pokec/"
        test_idx = False
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                               sens_attr,
                                                                                               predict_attr,
                                                                                               path=path,
                                                                                               label_number=label_number,
                                                                                               sens_number=sens_number,
                                                                                               seed=seed,
                                                                                               test_idx=test_idx)
        labels[labels > 1] = 1
    else:
        print('Invalid dataset name!!')
        exit(0)

    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
    features = features.to(args.device)
    num_nodes = features.shape[0]
    edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes), ).to(args.device)
    labels = labels.to(args.device)
    data = Data(edge_index, features, labels, idx_train, idx_val, idx_test, sens, sens_idx)

    # The number of classier
    num_class = 1

    """
    Build model and optimizer
    """
    # Synthetic teacher model and optimizer
    syn_t = SynTeacher(in_dim=features.shape[1], hid_dim=args.hidden, out_dim=num_class)
    optimizer_mlp = optim.Adam(syn_t.para_mlp, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_gnn = optim.Adam(syn_t.para_gnn, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_proj = optim.Adam(syn_t.para_proj, lr=args.lr, weight_decay=args.weight_decay)
    syn_t = syn_t.to(args.device)

    # Student model and optimizer
    if args.model == 'gcn':
        # student model
        stu_model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=num_class,
                    dropout=args.dropout)
        optimizer = optim.Adam(stu_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        stu_model = stu_model.to(args.device)

        # trained GNN model f_{cg} (for synthetic teacher)
        trained_gnn = GCN(nfeat=features.shape[1],
                          nhid=args.hidden,
                          nclass=num_class,
                          dropout=args.dropout)
        optimizer_van = optim.Adam(trained_gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        trained_gnn = trained_gnn.to(args.device)

    elif args.model == 'gin':
        stu_model = GIN(nfeat=features.shape[1],
                        nhid=args.hidden,
                        nclass=num_class,
                        dropout=args.dropout)
        optimizer = optim.Adam(stu_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        stu_model = stu_model.to(args.device)

        # trained GNN model f_{cg} (for synthetic teacher)
        trained_gnn = GIN(nfeat=features.shape[1],
                          nhid=args.hidden,
                          nclass=num_class,
                          dropout=args.dropout)
        optimizer_van = optim.Adam(trained_gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        trained_gnn = trained_gnn.to(args.device)

    """
    Train model (Synthetic teacher model and Student model)
    """
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_cont = ContLoss(tem=args.tem)

    # train f_{cg}
    train_vanilla(trained_gnn, optimizer_van, criterion_bce, args.epochs, data, save_name=f'{args.model}_vanilla.pt')

    # obtain node representations from the trained GNN model
    trained_gnn.load_state_dict(torch.load(f'{args.model}_vanilla.pt'))
    trained_gnn.eval()
    with torch.no_grad():
        h_van, output_van = trained_gnn(features, edge_index.to(args.device))

    # fairness experts training
    h_fair_mlp = syn_t.train_expert_mlp(optimizer=optimizer_mlp, criterion=criterion_bce, epochs=args.epochs, data=data)
    h_fair_gnn = syn_t.train_expert_gnn(optimizer=optimizer_gnn, criterion=criterion_bce, epochs=args.epochs, data=data)

    # projector training
    proj_input = torch.cat((h_fair_mlp, h_fair_gnn), 1)
    h_fair = syn_t.train_projector(optimizer=optimizer_proj, criterion=criterion_cont, epochs=args.epochs, data=data,
                                   input=proj_input, label=h_van)

    # student model training (knowledge distillation)
    train_student(stu_model, optimizer, criterion_bce, criterion_cont, args, data, save_name=f'{args.model}_student.pt',
                  soft_target=h_fair)

    """
    evaluation
    """
    auc, f1, acc, dp, eo = evaluation(stu_model, f'{args.model}_student.pt', data)

    return auc, f1, acc, dp, eo


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=0, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--proj_hidden', type=int, default=16,
                        help='Number of hidden units in the projection layer of encoder.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='loan',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'])
    parser.add_argument("--num_heads", type=int, default=1, help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gin'])
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--tem', type=float, default=0.5, help='the temperature of contrastive learning loss '
                                                               '(mutual information maximize)')
    parser.add_argument('--gamma', type=float, default=0.25, help='empower coefficient')
    parser.add_argument('--lr_w', type=float, default=1,
                        help='the learning rate of the adaptive weight coefficient')

    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    auc, f1, acc, dp, eo = np.zeros(shape=(args.seed_num, 2)), np.zeros(shape=(args.seed_num, 2)), \
                                     np.zeros(shape=(args.seed_num, 2)), np.zeros(shape=(args.seed_num, 2)), \
                                     np.zeros(shape=(args.seed_num, 2))

    for seed in range(args.seed_num):
        # set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)

        # torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        auc[seed, :], f1[seed, :], acc[seed, :], dp[seed, :], eo[seed, :] = run(args)

        print(f"========seed {seed}========")

    # print report
    print("=================START=================")
    print(f"Parameter:τ={args.tem}, γ={args.gamma}, lr_w={args.lr_w}")
    print(f"AUCROC: {np.around(np.mean(auc[:, 0]) * 100, 2)} ± {np.around(np.std(auc[:, 0]) * 100, 2)}")
    print(f'F1-score: {np.around(np.mean(f1[:, 0]) * 100, 2)} ± {np.around(np.std(f1[:, 0]) * 100, 2)}')
    print(f'ACC: {np.around(np.mean(acc[:, 0]) * 100, 2)} ± {np.around(np.std(acc[:, 0]) * 100, 2)}')
    print(f'parity: {np.around(np.mean(dp[:, 0]) * 100, 2)} ± {np.around(np.std(dp[:, 0]) * 100, 2)}')
    print(f'Equality: {np.around(np.mean(eo[:, 0]) * 100, 2)} ± {np.around(np.std(eo[:, 0]) * 100, 2)}')
    print("=================END=================")

