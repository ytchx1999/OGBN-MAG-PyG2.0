# OGBN-MAG-PyG2.0-Learn

学习PyG2.0异构图的使用。

## Environment Setup
```bash
(CentOS8, cuda == 10.2)
torch == 1.8.2
pyg == 2.0.1
```
### conda install
```bash
conda create -n env_pyg2 python=3.7
source activate env_pyg2
# pytorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
# pyg
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.8.0+cu101.html
# ogb
pip install ogb
```
## Experiment setup
```bash
cd src/
python main.py
```

## Reference：
+ [Docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html#)
+ [Code](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py)

## Download：
+ [OGBN-MAG](http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip)
+ [Metapath2vec Embedding (800+MB)](https://data.pyg.org/datasets/mag_metapath2vec_emb.zip)
+ [TransE Embedding (1+GB)](https://data.pyg.org/datasets/mag_transe_emb.zip) 

## Model
```python
GraphModule(
  (convs): ModuleList(
    (0): ModuleDict(
      (author__affiliated_with__institution): GATConv((-1, -1), 128, heads=4)
      (author__writes__paper): GATConv((-1, -1), 128, heads=4)
      (paper__cites__paper): GATConv((-1, -1), 128, heads=4)
      (paper__has_topic__field_of_study): GATConv((-1, -1), 128, heads=4)
      (institution__rev_affiliated_with__author): GATConv((-1, -1), 128, heads=4)
      (paper__rev_writes__author): GATConv((-1, -1), 128, heads=4)
      (field_of_study__rev_has_topic__paper): GATConv((-1, -1), 128, heads=4)
    )
    (1): ModuleDict(
      (author__affiliated_with__institution): GATConv((-1, -1), 128, heads=4)
      (author__writes__paper): GATConv((-1, -1), 128, heads=4)
      (paper__cites__paper): GATConv((-1, -1), 128, heads=4)
      (paper__has_topic__field_of_study): GATConv((-1, -1), 128, heads=4)
      (institution__rev_affiliated_with__author): GATConv((-1, -1), 128, heads=4)
      (paper__rev_writes__author): GATConv((-1, -1), 128, heads=4)
      (field_of_study__rev_has_topic__paper): GATConv((-1, -1), 128, heads=4)
    )
  )
  (lins): ModuleList(
    (0): ModuleDict(
      (paper): Linear(-1, 512, bias=True)
      (author): Linear(-1, 512, bias=True)
      (institution): Linear(-1, 512, bias=True)
      (field_of_study): Linear(-1, 512, bias=True)
    )
    (1): ModuleDict(
      (paper): Linear(-1, 512, bias=True)
      (author): Linear(-1, 512, bias=True)
      (institution): Linear(-1, 512, bias=True)
      (field_of_study): Linear(-1, 512, bias=True)
    )
  )
  (bns): ModuleList(
    (0): ModuleDict(
      (paper): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (author): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (institution): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (field_of_study): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): ModuleDict(
      (paper): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (author): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (institution): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (field_of_study): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (fc_out): ModuleDict(
    (paper): Linear(-1, 349, bias=True)
    (author): Linear(-1, 349, bias=True)
    (institution): Linear(-1, 349, bias=True)
    (field_of_study): Linear(-1, 349, bias=True)
  )
)
```

## Tree

```bash
.
├── data
│   └── mag
│       ├── processed
│       │   ├── data_metapath2vec.pt
│       │   ├── data_transe.pt
│       │   ├── pre_filter.pt
│       │   └── pre_transform.pt
│       └── raw
│           ├── mag_metapath2vec_emb.pt
│           ├── mag_transe_emb.pt
│           ├── node-feat
│           │   └── paper
│           │       ├── node-feat.csv.gz
│           │       └── node_year.csv.gz
│           ├── node-label
│           │   └── paper
│           │       └── node-label.csv.gz
│           ├── num-node-dict.csv.gz
│           ├── relations
│           │   ├── author___affiliated_with___institution
│           │   │   ├── edge.csv.gz
│           │   │   ├── edge_reltype.csv.gz
│           │   │   └── num-edge-list.csv.gz
│           │   ├── author___writes___paper
│           │   │   ├── edge.csv.gz
│           │   │   ├── edge_reltype.csv.gz
│           │   │   └── num-edge-list.csv.gz
│           │   ├── paper___cites___paper
│           │   │   ├── edge.csv.gz
│           │   │   ├── edge_reltype.csv.gz
│           │   │   └── num-edge-list.csv.gz
│           │   └── paper___has_topic___field_of_study
│           │       ├── edge.csv.gz
│           │       ├── edge_reltype.csv.gz
│           │       └── num-edge-list.csv.gz
│           └── split
│               └── time
│                   ├── nodetype-has-split.csv.gz
│                   └── paper
│                       ├── test.csv.gz
│                       ├── train.csv.gz
│                       └── valid.csv.gz
├── ogbn_mag2.ipynb
├── ogbn_mag.ipynb
├── README.md
├── src
│   ├── __init__.py
│   ├── main.py
│   └── models
│       ├── __init__.py
│       └── model.py
└── test_data.py
```