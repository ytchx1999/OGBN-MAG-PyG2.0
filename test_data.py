import argparse
import os.path as osp
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import ReLU
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader, HGTLoader
from torch_geometric.nn import Sequential, SAGEConv, Linear, to_hetero, TransformerConv, GATConv, Linear, HeteroConv

parser = argparse.ArgumentParser()
parser.add_argument('--use_hgt_loader', action='store_true')
args = parser.parse_args()

# path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/OGB')
transform = T.ToUndirected(merge=True)
dataset = OGB_MAG(root='./data', preprocess='metapath2vec',
                  transform=transform)
data = dataset[0]

transe_emb = torch.load('./data/mag/raw/mag_transe_emb.pt', map_location='cpu')
# print(transe_emb['paper'].shape)
# print(transe_emb['author'].shape)
data['paper'].x = torch.cat([data['paper'].x, transe_emb['paper']], dim=1)
data['author'].x = torch.cat([data['author'].x, transe_emb['author']], dim=1)
data['field_of_study'].x = torch.cat(
    [data['field_of_study'].x, transe_emb['field_of_study']], dim=1)
data['institution'].x = torch.cat(
    [data['institution'].x, transe_emb['institution']], dim=1)

train_input_nodes = ('paper', data['paper'].train_mask)
val_input_nodes = ('paper', data['paper'].val_mask)
test_input_nodes = ('paper', data['paper'].test_mask)
kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}

if not args.use_hgt_loader:
    train_loader = NeighborLoader(data, num_neighbors=[10] * 2, shuffle=True,
                                  input_nodes=train_input_nodes, **kwargs)
    val_loader = NeighborLoader(data, num_neighbors=[10] * 2,
                                input_nodes=val_input_nodes, **kwargs)
    test_loader = NeighborLoader(data, num_neighbors=[10] * 2,
                                 input_nodes=test_input_nodes, **kwargs)
else:
    train_loader = HGTLoader(data, num_samples=[1024] * 4, shuffle=True,
                             input_nodes=train_input_nodes, **kwargs)
    val_loader = HGTLoader(data, num_samples=[1024] * 4,
                           input_nodes=val_input_nodes, **kwargs)
    test_loader = HGTLoader(data, num_samples=[1024] * 4,
                            input_nodes=test_input_nodes, **kwargs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# model = Sequential('x, edge_index', [
#     (GATConv((-1, -1), 128, heads=3), 'x, edge_index -> x'),
#     ReLU(inplace=True),
#     (GATConv((-1, -1), 128, heads=3), 'x, edge_index -> x'),
#     ReLU(inplace=True),
#     (Linear(-1, dataset.num_classes), 'x -> x'),
# ])


class Net1(torch.nn.Module):
    def __init__(self, hidden_dim, num_classes, num_layers=2) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(self.num_layers):
            self.convs.append(GATConv((-1, -1), hidden_dim, heads=4))
            self.lins.append(Linear(-1, hidden_dim*4))
            self.bns.append(nn.BatchNorm1d(hidden_dim*4))

        # self.dropout = torch.nn.Dropout()
        self.fc_out = Linear(-1, num_classes)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index) + self.lins[i](x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc_out(x)
        return x


class Net2(nn.Module):
    def __init__(self, metadata, hidden_dim, num_classes, num_layers=2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()

        for i in range(self.num_layers):
            self.convs.append(HeteroConv({
                ('author', 'affiliated_with', 'institution'): SAGEConv((-1, -1), hidden_dim),
                ('author', 'writes', 'paper'): GATConv((-1, -1), hidden_dim),
                ('paper', 'cites', 'paper'): TransformerConv((-1, -1), hidden_dim),
                ('paper', 'has_topic', 'field_of_study'): GATConv((-1, -1), hidden_dim),
                ('institution', 'rev_affiliated_with', 'author'): SAGEConv((-1, -1), hidden_dim),
                ('paper', 'rev_writes', 'author'): SAGEConv((-1, -1), hidden_dim),
                ('field_of_study', 'rev_has_topic', 'paper'): SAGEConv((-1, -1), hidden_dim)
            }, aggr='sum'))

            self.lins.append(Linear(-1, hidden_dim))

        self.fc_out = Linear(-1, num_classes)

    def forward(self, x_dict, edge_index_dict):
        for i in range(self.num_layers):
            x_linear = {key: self.lins[i](x) for key, x in x_dict.items()}
            x_dict = self.convs[i](x_dict, edge_index_dict)
            x_dict = {key: x + x_linear[key] for key, x in x_dict.items()}
            x_dict = {key: nn.BatchNorm1d(x.shape[1]).to(device)(
                x) for key, x in x_dict.items()}
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=0.5, training=self.training)
                      for key, x in x_dict.items()}

        x_dict['paper'] = self.fc_out(x_dict['paper'])
        return x_dict


model = Net1(hidden_dim=128, num_classes=dataset.num_classes, num_layers=2)
model = to_hetero(model, data.metadata(), aggr='sum').to(device)

# model = Net2(data.metadata(), hidden_dim=64,
#              num_classes=dataset.num_classes, num_layers=2)
# model.to(device)


@torch.no_grad()
def init_params():
    # Initialize lazy parameters via forwarding a single batch to the model:
    batch = next(iter(train_loader))
    batch = batch.to(device)
    model(batch.x_dict, batch.edge_index_dict)


def train():
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
        loss = F.cross_entropy(out, batch['paper'].y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    total_examples = total_correct = 0
    for batch in tqdm(loader):
        batch = batch.to(device)
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
        pred = out.argmax(dim=-1)

        total_examples += batch_size
        total_correct += int((pred == batch['paper'].y[:batch_size]).sum())

    return total_correct / total_examples


init_params()  # Initialize parameters.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 21):
    loss = train()
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    print(
        f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test:{test_acc:.4f}')
