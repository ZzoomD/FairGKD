import os
import dgl
import torch
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import distance_matrix
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch.optim as optim


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh*max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    # print('building edge relationship complete')
    idx_map =  np.array(idx_map)
    
    return idx_map


def load_pokec(dataset, sens_attr, predict_attr, path="./datasets/pokec/", label_number=1000, sens_number=500,
               seed=19, test_idx=False):
    """Load data"""
    # print('Loading {} dataset from {}'.format(dataset, path))

    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    # header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)), label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]

    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


def load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="./dataset/bail/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    # header.remove(sens_attr)
    
    # # Normalize School
    # idx_features_labels['SCHOOL'] = 2*(idx_features_labels['SCHOOL']-idx_features_labels['SCHOOL'].min()).div(idx_features_labels['SCHOOL'].max() - idx_features_labels['SCHOOL'].min()) - 1

    # # Normalize RULE
    # idx_features_labels['RULE'] = 2*(idx_features_labels['RULE']-idx_features_labels['RULE'].min()).div(idx_features_labels['RULE'].max() - idx_features_labels['RULE'].min()) - 1

    # # Normalize AGE
    # idx_features_labels['AGE'] = 2*(idx_features_labels['AGE']-idx_features_labels['AGE'].min()).div(idx_features_labels['AGE'].max() - idx_features_labels['AGE'].min()) - 1

    # # Normalize TSERVD
    # idx_features_labels['TSERVD'] = 2*(idx_features_labels['TSERVD']-idx_features_labels['TSERVD'].min()).div(idx_features_labels['TSERVD'].max() - idx_features_labels['TSERVD'].min()) - 1

    # # Normalize FOLLOW
    # idx_features_labels['FOLLOW'] = 2*(idx_features_labels['FOLLOW']-idx_features_labels['FOLLOW'].min()).div(idx_features_labels['FOLLOW'].max() - idx_features_labels['FOLLOW'].min()) - 1

    # # Normalize TIME
    # idx_features_labels['TIME'] = 2*(idx_features_labels['TIME']-idx_features_labels['TIME'].min()).div(idx_features_labels['TIME'].max() - idx_features_labels['TIME'].min()) - 1

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    return adj, features, labels, idx_train, idx_val, idx_test, sens


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1


def train_vanilla(model, optimizer, criterion, epochs, data, save_name):
    best_loss = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        h, output = model(data.features, data.edge_index)
        loss_train = criterion(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
        loss_train.backward()
        optimizer.step()

        model.eval()
        h, output = model(data.features, data.edge_index)
        loss_val = criterion(output[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float())

        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
            torch.save(model.state_dict(), save_name)


def train_student(model, optimizer, criterion_bce, criterion_kd, args, data, save_name, soft_target):
    best_loss = 100
    best_result = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        h, output = model(data.features, data.edge_index)

        loss_bce_train = criterion_bce(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
        loss_kd_train = criterion_kd(h[data.idx_train], soft_target[data.idx_train])
        if epoch == 0:
            weight_compute = AdaWeight(loss_bce_train, loss_kd_train, lr=args.lr_w, gamma=args.gamma)
        lad1, lad2 = weight_compute.compute(loss_bce_train.item(), loss_kd_train.item())
        loss_train = lad1 * loss_bce_train + lad2 * loss_kd_train

        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            h, output = model(data.features, data.edge_index)
        loss_bce_val = criterion_bce(output[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float())
        loss_kd_val = criterion_kd(h[data.idx_val], soft_target[data.idx_val])
        loss_val = loss_bce_val + loss_kd_val
        output_preds = (output.squeeze() > 0).type_as(data.labels)
        acc = accuracy_score(data.labels[data.idx_val].cpu().numpy(), output_preds[data.idx_val].cpu().numpy())
        parity, equality = fair_metric(output_preds[data.idx_val].cpu().numpy(),
                                       data.labels[data.idx_val].cpu().numpy(),
                                       data.sens[data.idx_val].numpy())

        if args.dataset == 'pokec_z' and args.model == 'gin':
            if acc - 3*(parity + equality) > best_result:
                best_result = acc - 3*(parity + equality)
                torch.save(model.state_dict(), save_name)
        else:
            if loss_val.item() < best_loss:
                best_loss = loss_val.item()
                torch.save(model.state_dict(), save_name)
                # print(f"[Train] Epoch {epoch}:bce_loss: {loss_bce_train.item():.4f} | kd_loss: {loss_kd_train.item():.4f} "
                #       f"| total_loss: {loss_train.item():.4f} | lad1: {lad1:.4f}, lad2: {lad2:.4f}")


def evaluation(model, weight_path, data):
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    with torch.no_grad():
        h, output = model(data.features, data.edge_index)
    output_preds = (output.squeeze() > 0).type_as(data.labels)
    auc_test = roc_auc_score(data.labels.cpu().numpy()[data.idx_test.cpu()],
                                 output.detach().cpu().numpy()[data.idx_test.cpu()])
    f1_test = f1_score(data.labels[data.idx_test].cpu().numpy(), output_preds[data.idx_test].cpu().numpy())
    acc_test = accuracy_score(data.labels[data.idx_test].cpu().numpy(), output_preds[data.idx_test].cpu().numpy())
    dp_test, eo_test = fair_metric(output_preds[data.idx_test].cpu().numpy(), data.labels[data.idx_test].cpu().numpy(),
                                     data.sens[data.idx_test].numpy())
    return auc_test, f1_test, acc_test, dp_test, eo_test


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()


class AdaWeight:
    def __init__(self, loss1_init, loss2_init, weight_loss1=0.5, lr=0.025, gamma=0.25):
        self.loss1_init = loss1_init
        self.loss2_init = loss2_init
        self.weight_loss1 = weight_loss1
        self.lr = lr
        self.gamma = gamma

    def compute(self, loss1, loss2):
        rela_loss1 = (loss1 / self.loss1_init.item())**self.gamma
        rela_loss2 = (loss2 / self.loss2_init.item())**self.gamma
        rela_weight_loss1 = rela_loss1 / (rela_loss1 + rela_loss2)
        self.weight_loss1 = self.lr * rela_weight_loss1 + (1 - self.lr) * self.weight_loss1
        self.weight_loss2 = 1 - self.weight_loss1
        return self.weight_loss1, self.weight_loss2


class ContLoss(_Loss):
    def __init__(self, reduction='mean', tem: float=0.5):
        super(ContLoss, self).__init__()
        self.reduction = reduction
        self.tem: float = tem

    def sim(self, h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return torch.mm(h1, h2.t())

    def loss(self, h1: torch.Tensor, h2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tem)
        intra_sim = f(self.sim(h1, h1))
        inter_sim = f(self.sim(h1, h2))
        return -torch.log(inter_sim.diag() / (inter_sim.sum(1) + intra_sim.sum(1) - intra_sim.diag()))

    def forward(self, h1: torch.Tensor, h2: torch.Tensor):
        l1 = self.loss(h1, h2)
        l2 = self.loss(h2, h1)
        ret = (l1 + l2) / 0.5
        ret = ret.mean() if self.reduction=='mean' else ret.sum()
        return ret
