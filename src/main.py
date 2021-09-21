from pickle import load
import torch_geometric
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import to_hetero
import torch_geometric.transforms as T

import argparse
import os
import tqdm

from models.model import GAT


def load_data(args):
    dataset = OGB_MAG(root=args.dataset_dir, preprocess='metapath2vec', transform=T.ToUndirected(merge=True))
    data = dataset[0]

    metapath_emb = torch.load(args.metapath_embed_dir, map_location='cpu')
    data['paper'].x = torch.cat([data['paper'].x, metapath_emb['paper']], dim=1)

    if not os.path.exists(args.transe_embed_dir):
        _ = OGB_MAG(root=args.dataset_dir, preprocess='transe', transform=T.ToUndirected(merge=True))

    transe_emb = torch.load(args.transe_embed_dir, map_location='cpu')
    data['paper'].x = torch.cat([data['paper'].x, transe_emb['paper']], dim=1)
    data['author'].x = torch.cat([data['author'].x, transe_emb['author']], dim=1)
    data['field_of_study'].x = torch.cat([data['field_of_study'].x, transe_emb['field_of_study']], dim=1)
    data['institution'].x = torch.cat([data['institution'].x, transe_emb['institution']], dim=1)

    return dataset, data


@torch.no_grad()
def init_params(model, train_loader, device):
    # Initialize lazy parameters via forwarding a single batch to the model:
    batch = next(iter(train_loader))
    batch = batch.to(device)
    model(batch.x_dict, batch.edge_index_dict)


def main():
    parser = argparse.ArgumentParser(description='R-GAT (OGBN-MAG)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_neighbors', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dataset_dir', type=str, default='../data/')
    parser.add_argument('--metapath_embed_dir', type=str, default='../data/mag/raw/mag_metapath2vec_emb.pt')
    parser.add_argument('--transe_embed_dir', type=str, default='../data/mag/raw/mag_transe_emb.pt')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=25)
    args = parser.parse_args()
    print(args)

    dataset, data = load_data(args)

    train_input_nodes = ('paper', data['paper'].train_mask)
    val_input_nodes = ('paper', data['paper'].val_mask)
    test_input_nodes = ('paper', data['paper'].test_mask)

    kwargs = {'batch_size': args.batch_size, 'num_workers': 6, 'persistent_workers': True}

    train_loader = NeighborLoader(data, num_neighbors=[args.num_neighbors, args.num_neighbors], input_nodes=train_input_nodes, shuffle=True, **kwargs)
    val_loader = NeighborLoader(data, num_neighbors=[args.num_neighbors, args.num_neighbors], input_nodes=val_input_nodes, **kwargs)
    test_loader = NeighborLoader(data, num_neighbors=[args.num_neighbors, args.num_neighbors], input_nodes=test_input_nodes, **kwargs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = GAT(hidden_dim=args.hidden_dim, num_classes=dataset.num_classes, heads=args.heads, num_layers=args.num_layers)
    model = to_hetero(model, data.metadata(), aggr='sum').to(device)

    init_params(model, train_loader, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        print(
            f'epoch: {epoch:02d}, '
            f'train_loss: {train_loss:.4f}, '
            f'train_acc: {train_acc:.4f}, '
            f'val_acc, {val_acc:.4f} '
            f'test_acc, {test_acc:.4f} '
        )


def train(model, train_loader, device, criterion, optimizer):
    model.train()

    tot_loss = 0
    tot_correct = 0
    tot_example = 0
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        batch_size = batch['paper'].batch_size

        out = model(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
        loss = criterion(out, batch['paper'].y[:batch_size])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot_loss += loss.item() * batch_size
        y_pred = out.argmax(dim=1)
        correct = (y_pred == batch['paper'].y[:batch_size]).sum().item()
        tot_correct += correct
        tot_example += batch_size

    return tot_loss / tot_example, tot_correct / tot_example


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()

    tot_loss = 0
    tot_correct = 0
    tot_example = 0
    for batch in tqdm(loader):
        batch = batch.to(device)
        batch_size = batch['paper'].batch_size

        out = model(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
        loss = criterion(out, batch['paper'].y[:batch_size])

        tot_loss += loss.item() * batch_size
        y_pred = out.argmax(dim=1)
        correct = (y_pred == batch['paper'].y[:batch_size]).sum().item()
        tot_correct += correct
        tot_example += batch_size

    return tot_loss / tot_example, tot_correct / tot_example


if __name__ == '__main__':
    main()
