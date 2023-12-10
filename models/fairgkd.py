import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Projector, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.lin3 = nn.Linear(out_dim, out_dim)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h):
        y = self.lin1(h)
        y = self.lin2(y)
        y = self.lin3(y)
        return y


class SynTeacher(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim=1):
        super(SynTeacher, self).__init__()
        self.expert_mlp = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                        nn.Linear(hid_dim, hid_dim))
        self.expert_gnn = GCNConv(in_dim, hid_dim)
        self.projector = Projector(2*hid_dim, hid_dim)

        # c1 and c2 serve as the classifier in fairness experts training
        self.c1 = nn.Linear(hid_dim, out_dim)
        self.c2 = nn.Linear(hid_dim, out_dim)

        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            self.weights_init(m)

        self.para_mlp = list(self.expert_mlp.parameters()) + list(self.c1.parameters())
        self.para_gnn = list(self.expert_gnn.parameters()) + list(self.c2.parameters())
        self.para_proj = list(self.projector.parameters())

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, x_ones):
        h1 = self.forward_mlp(x)
        h2 = self.expert_gnn(x_ones, edge_index)
        h = self.projector(torch.cat((h1, h2), 1))
        y = self.classifier(h)
        return h, y

    def distill(self, x, edge_index, x_ones):
        h1 = self.forward_mlp(x)
        h2 = self.expert_gnn(x_ones, edge_index)
        h = self.projector(torch.cat((h1, h2), 1))
        return h

    def forward_mlp(self, x):
        h = x
        for l, layer in enumerate(self.expert_mlp):
            h = layer(h)
            h = F.relu(h)
            h = self.dropout(h)
        y = self.c1(h)
        return h, y

    def forward_gnn(self, x_ones, edge_index):
        h = self.expert_gnn(x_ones, edge_index)
        y = self.c2(h)
        return h, y

    # fairness experts training consists of train_expert_mlp and train_expert_gnn
    def train_expert_mlp(self, optimizer, criterion, epochs, data):
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            h, output = self.forward_mlp(data.features)
            loss_train = criterion(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
            loss_train.backward()
            optimizer.step()

        self.eval()
        with torch.no_grad():
            h_mlp, output_mlp = self.forward_mlp(data.features)
        return h_mlp

    def train_expert_gnn(self, optimizer, criterion, epochs, data):
        features_one = torch.ones_like(data.features).to(data.features.device)
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            h, output = self.forward_gnn(features_one, data.edge_index)
            loss_train = criterion(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
            loss_train.backward()
            optimizer.step()

        self.eval()
        with torch.no_grad():
            h_gnn, output_gnn = self.forward_gnn(features_one, data.edge_index)
        return h_gnn

    def train_projector(self, optimizer, criterion, epochs, data, input, label):
        for epoch in range(epochs):
            # train projector
            self.train()
            optimizer.zero_grad()

            output_proj = self.projector(input)
            loss_train = criterion(output_proj[data.idx_train], label[data.idx_train])
            loss_train.backward()
            optimizer.step()

        self.eval()
        with torch.no_grad():
            h_proj = self.projector(input)
        return h_proj
