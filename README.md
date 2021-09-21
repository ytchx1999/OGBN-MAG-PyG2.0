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
```bash
data
HeteroData(
  paper={
    x=[736389, 512],
    y=[736389],
    train_mask=[736389],
    val_mask=[736389],
    test_mask=[736389]
  },
  author={ x=[1134649, 384] },
  institution={ x=[8740, 384] },
  field_of_study={ x=[59965, 384] },
  (author, affiliated_with, institution)={ edge_index=[2, 1043998] },
  (author, writes, paper)={ edge_index=[2, 7145660] },
  (paper, cites, paper)={ edge_index=[2, 10792672] },
  (paper, has_topic, field_of_study)={ edge_index=[2, 7505078] },
  (institution, rev_affiliated_with, author)={ edge_index=[2, 1043998] },
  (paper, rev_writes, author)={ edge_index=[2, 7145660] },
  (field_of_study, rev_has_topic, paper)={ edge_index=[2, 7505078] }
)
self
GAT(
  (convs): ModuleList(
    (0): GATConv((-1, -1), 128, heads=4)
  )
  (bns): ModuleList()
  (lins): ModuleList()
)
special variables:
function variables:
T_destination: ~T_destination
bns: ModuleList()
convs: ModuleList(
  (0): GATConv((-1, -1), 128, heads=4)
)
special variables:
function variables:
T_destination: ~T_destination
dump_patches: False
training: True
_apply: <bound method Module._apply of ModuleList(
  (0): GATConv((-1, -1), 128, heads=4)
)>
_backward_hooks: OrderedDict([])
_buffers: OrderedDict([])
_call_impl: <bound method Module._call_impl of ModuleList(
  (0): GATConv((-1, -1), 128, heads=4)
)>
_forward_hooks: OrderedDict([])
_forward_pre_hooks: OrderedDict([])
_get_abs_string_index: <bound method ModuleList._get_abs_string_index of ModuleList(
  (0): GATConv((-1, -1), 128, heads=4)
)>
_get_backward_hooks: <bound method Module._get_backward_hooks of ModuleList(
  (0): GATConv((-1, -1), 128, heads=4)
)>
_get_name: <bound method Module._get_name of ModuleList(
  (0): GATConv((-1, -1), 128, heads=4)
)>
_is_full_backward_hook: None
_load_from_state_dict: <bound method Module._load_from_state_dict of ModuleList(
  (0): GATConv((-1, -1), 128, heads=4)
)>
_load_state_dict_pre_hooks: OrderedDict([])
_maybe_warn_non_full_backward_hook: <bound method Module._maybe_warn_non_full_backward_hook of ModuleList(
  (0): GATConv((-1, -1), 128, heads=4)
)>
_modules: OrderedDict([('0', GATConv((-1, -1), 12..., heads=4))])
special variables:
function variables:
'0': GATConv((-1, -1), 128, heads=4)
special variables:
function variables:
T_destination: ~T_destination
add_self_loops: True
aggr: 'add'
att_dst: Parameter containing:
tensor([[[ 0.0503,  0.1882, -0.0239, -0.1180,  0.0186,  0.0712, -0.0430,
           0.1996, -0.1122,  0.0329,  0.1708,  0.0202, -0.0035,  0.0464,
           0.1158,  0.0426,  0.1526,  0.0780, -0.0124,  0.1335,  0.0974,
          -0.1333, -0.1973, -0.1556,  0.1013, -0.1580,  0.1979, -0.0292,
          -0.0966, -0.1001, -0.1911,  0.0158, -0.0019, -0.2098, -0.0763,
           0.0122, -0.0111, -0.1995, -0.1606,  0.0394,  0.1297,  0.0061,
           0.0226, -0.0059, -0.0693,  0.1859,  0.1181, -0.1136,  0.1884,
          -0.0206,  0.1602, -0.1944, -0.1559,  0.0762, -0.1625,  0.0470,
          -0.1268,  0.1659, -0.0949, -0.0066, -0.1111,  0.0562,  0.1007,
          -0.1198, -0.0542,  0.0302, -0.1336, -0.0920,  0.0406, -0.2045,
          -0.0614, -0.0801, -0.0738, -0.0112, -0.1674, -0.2066, -0.0216,
           0.0121, -0.2062,  0.1726, -0.0719, -0.1372, -0.0350,  0.1313,
          -0.0418, -0.0744, -0.0242, -0.1860,  0.0820, -0.1581, -0.1083,
          -0.1759, -0.1337,  0.0094,  0.2061,  0.0869,  0.0655, -0.1008,
           0.2074,  0.1536, -0.0055,  0.1756, -0.1168,  0.1607,  0.0467,
          -0.0810, -0.0710, -0.1037,  0.2111,  0.0024,  0.1892,  0.0871,
           0.0498,  0.0468,  0.1877,  0.1931, -0.2018,  0.0501,  0.1679,
           0.0506, -0.2025, -0.0670,  0.1697, -0.1773,  0.0347, -0.0368,
          -0.0951,  0.0690],
         [ 0.1122,  0.1434, -0.1681, -0.0105,  0.2043,  0.1924, -0.0332,
          -0.0789,  0.0439, -0.1078, -0.0763, -0.1924,  0.0564, -0.1540,
          -0.1965, -0.0709,  0.0377,  0.1885,  0.0949,  0.1907,  0.0934,
          -0.0984,  0.2095,  0.1492,  0.1422, -0.1030, -0.0556,  0.1935,
          -0.0581,  0.1237, -0.1633,  0.0172,  0.1032,  0.0694,  0.0731,
          -0.0186,  0.1318, -0.0832,  0.1999,  0.1571,  0.0011, -0.1431,
           0.0464, -0.0188, -0.1608,  0.0051, -0.0210, -0.0306,  0.0574,
           0.1704, -0.1861, -0.1559,  0.0681, -0.1097, -0.0543,  0.1149,
          -0.0509, -0.1534, -0.0177, -0.1782,  0.0382,  0.0584, -0.0685,
           0.0735, -0.0841,  0.2118, -0.1661, -0.0514, -0.0242,  0.0012,
          -0.0803, -0.2030,  0.1436,  0.1847,  0.1129,  0.1168, -0.1511,
           0.1379, -0.1920,  0.1983, -0.2083, -0.1514,  0.2114,  0.0540,
          -0.0640, -0.0914, -0.1922,  0.2092, -0.2064, -0.2098,  0.1906,
           0.0523,  0.1772, -0.1785, -0.1664, -0.1169,  0.1927, -0.1091,
           0.0658,  0.1018, -0.0613,  0.1411, -0.0786,  0.0514, -0.0391,
          -0.1273,  0.1660, -0.1239,  0.0545,  0.1809,  0.0130, -0.1136,
          -0.0184, -0.0039, -0.1975, -0.0603,  0.0182,  0.0930,  0.2049,
          -0.2070, -0.1231,  0.0516,  0.1974, -0.1321, -0.0786, -0.0168,
          -0.0095, -0.1326],
         [ 0.0685, -0.0016,  0.1161, -0.0167, -0.1468, -0.1682,  0.1360,
           0.2069,  0.0232,  0.0543,  0.2109,  0.2024, -0.0012,  0.1544,
          -0.1718, -0.0469,  0.0205, -0.1542, -0.1046, -0.1587,  0.0512,
          -0.0142, -0.0519,  0.0751, -0.0702,  0.1357,  0.1655,  0.1894,
           0.0215, -0.0217,  0.0120, -0.2004, -0.1623,  0.0518,  0.0463,
          -0.0795,  0.1615, -0.1630, -0.0089,  0.1850,  0.0292, -0.1822,
          -0.1223,  0.1639, -0.1269, -0.1694, -0.1458,  0.1720, -0.1188,
          -0.0324, -0.0829,  0.1883, -0.1489, -0.1713, -0.1200,  0.1856,
           0.1563,  0.1798,  0.0791, -0.1816,  0.1524,  0.0949, -0.0395,
           0.0178,  0.0866,  0.0126, -0.0193, -0.1655,  0.0496, -0.1990,
          -0.1691, -0.0546, -0.0350, -0.2015,  0.1711,  0.0200, -0.0355,
          -0.1782,  0.0427, -0.1671, -0.0254, -0.1123, -0.0849, -0.0877,
           0.1102,  0.0074, -0.0994,  0.1124,  0.1130,  0.0291, -0.0514,
          -0.0213, -0.1535,  0.1601, -0.1723, -0.0113,  0.0617, -0.0633,
          -0.0623,  0.1821, -0.0641,  0.0133,  0.2064,  0.0235, -0.1924,
          -0.2132,  0.1920,  0.0389, -0.0752, -0.0091,  0.0735,  0.1512,
          -0.0484,  0.0405,  0.0668, -0.0679, -0.0286,  0.1324, -0.1923,
          -0.1220,  0.2043,  0.0715,  0.0074, -0.1898,  0.0438,  0.1910,
           0.0791,  0.1274],
         [-0.1734, -0.0831,  0.1528,  0.0535, -0.0895, -0.1039,  0.0273,
           0.0334,  0.1962,  0.1805,  0.0682,  0.0006,  0.0261, -0.1889,
          -0.0589,  0.0916, -0.1367, -0.0821,  0.0175, -0.0296, -0.1291,
          -0.0276,  0.1176, -0.0593, -0.1107,  0.2001,  0.1217,  0.0880,
           0.0160,  0.0532,  0.2087, -0.1489, -0.0891, -0.2074,  0.1628,
          -0.0207,  0.0684, -0.0300,  0.1233,  0.0340,  0.1849,  0.1164,
           0.1754, -0.0122, -0.0511,  0.1646,  0.0506, -0.0055, -0.1902,
           0.0703, -0.0124, -0.2017, -0.0930, -0.0300,  0.2015, -0.1445,
          -0.1128, -0.1864,  0.1268, -0.0977, -0.0469,  0.1195,  0.0997,
          -0.0616,  0.0426,  0.0265, -0.1767, -0.0545,  0.1113,  0.1012,
           0.2050,  0.1910,  0.1745,  0.1568, -0.0551, -0.2032, -0.1489,
           0.1905,  0.1618, -0.0928,  0.1786,  0.0034, -0.0173,  0.0287,
           0.0215, -0.1032,  0.0715, -0.0629, -0.1857,  0.0900,  0.1152,
           0.0958, -0.1265,  0.1819, -0.0355,  0.1593,  0.1129, -0.0260,
           0.1629,  0.0023,  0.1381, -0.1628, -0.0519,  0.0217, -0.0740,
          -0.0291, -0.0297, -0.2075, -0.0960,  0.0653, -0.1065,  0.0995,
          -0.0819, -0.0504,  0.0989,  0.1978, -0.0075,  0.1619, -0.0780,
          -0.2015,  0.1817,  0.0978, -0.0708,  0.1752,  0.2115,  0.1241,
          -0.0068, -0.0322]]], requires_grad=True)
att_src: Parameter containing:
tensor([[[ 3.4111e-02,  1.6835e-01,  6.5422e-02,  2.0217e-01, -1.3026e-02,
          -1.2326e-01,  2.0219e-01, -1.9211e-01, -1.8400e-02,  9.7664e-02,
           1.3266e-01, -1.7304e-01,  2.0257e-01, -2.9001e-02,  4.6297e-02,
           5.0685e-02,  7.1423e-02, -2.0647e-01,  1.9492e-01,  1.0743e-02,
           1.2808e-01, -1.3745e-01,  6.0799e-02,  1.7980e-01, -8.6512e-02,
           1.4996e-02,  4.3405e-02, -4.7659e-02,  3.2015e-02,  1.3363e-01,
          -1.1351e-01,  1.4261e-01, -6.9108e-02, -1.5592e-01, -6.9176e-02,
          -2.0014e-01,  2.0460e-01,  1.8792e-01,  7.5818e-02, -1.0246e-01,
          -8.2347e-02, -7.7248e-02,  1.0634e-01,  1.9730e-01,  1.7788e-03,
           2.0051e-01, -8.7770e-02, -1.9851e-02,  1.5062e-01,  2.0759e-01,
          -1.7020e-02,  5.2160e-02,  7.3031e-02, -1.0342e-02,  1.7440e-01,
          -7.8850e-02,  2.4218e-02,  1.4064e-01, -1.2920e-01,  1.2601e-01,
          -8.8688e-02,  1.6203e-01, -8.1108e-02,  6.0323e-02,  1.5454e-02,
           1.8850e-01, -5.4309e-02, -1.8148e-01,  2.7970e-02, -2.8577e-02,
           1.2906e-01, -7.7467e-03, -7.4098e-02,  6.8699e-02, -1.6870e-01,
          -8.7056e-02,  1.5976e-01, -1.3471e-01, -1.4141e-01, -4.0735e-02,
           1.3838e-02, -2.1105e-01,  1.7038e-02, -1.1215e-03, -3.6576e-02,
          -9.8198e-02, -2.7037e-02,  1.5902e-01,  4.2706e-02, -1.9138e-01,
           3.4231e-02,  1.7150e-01,  1.1897e-02,  1.0338e-01, -1.5067e-01,
          -1.3223e-01, -1.9853e-01, -1.7823e-01,  1.1295e-01, -8.0096e-02,
          -9.1446e-02,  4.3991e-02,  1.9716e-01, -1.6489e-01,  1.6102e-01,
           2.5316e-02,  6.0883e-03, -3.5503e-02, -1.6143e-01,  4.6426e-03,
           1.7153e-01,  1.5239e-01, -8.2019e-02,  1.4203e-01,  2.1064e-01,
           1.0698e-01,  1.1502e-01,  9.3153e-02,  2.1065e-01, -4.8207e-02,
          -8.6228e-02, -5.2611e-02,  2.0737e-01,  4.9581e-02, -1.2241e-01,
           1.1991e-01, -1.3785e-01,  4.1180e-02],
         [-1.2195e-01, -3.5035e-02,  1.2843e-01, -4.0504e-02,  1.4302e-01,
          -6.6931e-02,  2.0487e-01,  1.6690e-01,  1.7537e-01,  1.4552e-01,
          -1.1628e-01,  7.4747e-02,  9.8418e-02, -5.0906e-02, -1.4222e-01,
           1.3107e-02,  8.0594e-02, -1.7702e-01,  1.9585e-01,  5.3347e-02,
          -1.3099e-01, -1.2589e-01,  4.2540e-02, -4.4332e-02,  1.8777e-01,
           8.7050e-02,  1.8132e-01,  1.8466e-01, -1.8968e-01, -1.0684e-01,
           1.9865e-01, -1.6549e-01, -5.1159e-02,  9.7090e-02, -1.9288e-01,
          -1.3730e-01, -4.0612e-02, -5.8398e-03, -1.1921e-01, -9.6399e-02,
          -3.4960e-03, -4.6413e-02, -1.5982e-01,  7.9141e-03, -1.9811e-01,
          -7.6714e-02,  1.0338e-01, -1.0337e-02,  6.1975e-03, -5.0126e-02,
          -1.4812e-01,  4.8058e-02,  2.0608e-01, -4.0107e-02, -1.2914e-01,
           4.7060e-02,  1.2474e-01,  3.7087e-02, -1.3302e-01,  1.0899e-01,
          -1.8340e-01, -5.8819e-02,  1.2753e-01, -2.1002e-01, -9.3865e-02,
          -1.8457e-01,  5.3072e-02, -5.5329e-03, -2.8928e-02, -1.3524e-01,
          -1.1962e-01,  2.8072e-02, -1.6206e-01, -9.0637e-02,  1.5703e-01,
          -9.0618e-02,  9.7052e-02,  2.0166e-02,  1.6610e-01,  4.1133e-02,
          -1.9640e-01, -1.8732e-01,  2.1200e-01, -1.9243e-01,  4.5247e-02,
          -1.8883e-01, -6.0365e-02,  1.9647e-01, -4.7599e-02,  1.8863e-02,
           8.6468e-02, -1.0887e-01,  1.1942e-01,  1.5392e-01, -6.4747e-02,
          -8.2486e-02, -5.3382e-02,  5.5889e-02, -1.0247e-01,  1.2534e-01,
           1.2355e-01,  4.8317e-02,  1.2243e-01,  1.2292e-01,  1.3865e-02,
           1.4450e-02,  1.1284e-01, -3.4176e-02, -1.9990e-01,  4.9685e-02,
          -9.2063e-03,  1.1429e-01,  9.3348e-02,  7.1035e-02,  1.4651e-01,
           5.8673e-03, -6.2074e-02, -6.8356e-02, -1.7816e-01, -2.0675e-02,
           9.3722e-03, -1.1962e-01, -1.2211e-01, -9.4426e-02,  1.9495e-01,
          -1.1972e-02, -8.9550e-02,  2.1246e-01],
         [ 1.5124e-01,  2.9009e-02, -1.5505e-01,  1.4252e-01, -5.8029e-02,
          -7.9111e-02,  1.7367e-01,  1.9363e-01, -6.3802e-02,  1.6547e-01,
          -1.5203e-02, -1.9645e-01, -7.0082e-02, -7.3699e-02, -6.0538e-02,
          -4.4670e-02,  1.4130e-01,  8.6645e-02,  1.8296e-02, -1.5295e-01,
           1.1160e-01,  8.3738e-02, -2.1248e-01,  4.7728e-02, -2.0453e-01,
          -1.8741e-01, -4.3314e-02, -1.5173e-01,  1.2730e-01,  6.4793e-02,
           1.2618e-01,  7.6348e-02, -1.9143e-01,  4.1113e-02, -8.4266e-02,
           7.2583e-02,  7.8952e-02,  1.3439e-01, -1.7924e-01, -9.4959e-03,
           2.1047e-01, -1.2000e-01,  9.9612e-02, -1.2964e-01, -9.1372e-02,
           6.1828e-02,  8.0444e-02, -1.1654e-01, -1.0917e-01, -1.1729e-02,
          -1.6245e-01, -1.5451e-02, -1.3593e-01,  1.7387e-04,  1.3003e-01,
          -2.1161e-01, -2.0248e-02, -1.8730e-01, -6.0896e-02,  8.3339e-02,
           1.1773e-01, -1.5315e-01,  7.6742e-02,  1.2738e-01,  1.1625e-01,
          -1.7452e-01, -1.1202e-01, -1.9689e-01, -4.9102e-02,  1.5147e-01,
          -3.0540e-02,  7.6393e-02, -9.1213e-02, -8.6795e-02,  1.0008e-01,
          -1.4243e-01,  1.0205e-01, -1.8128e-01, -1.6144e-01,  5.4296e-02,
           2.1108e-01,  6.5756e-02,  7.3812e-02,  1.8385e-01, -1.6298e-01,
          -5.7162e-02,  5.0310e-02, -1.2530e-01, -7.5441e-02,  7.7820e-04,
           7.1807e-02, -1.0508e-01,  1.6336e-01, -1.7435e-01, -7.6788e-03,
          -2.0068e-01,  1.7495e-01,  5.8403e-02, -8.9610e-02,  1.4383e-01,
          -1.9828e-01, -1.7406e-01,  1.4034e-01,  8.8284e-02,  6.2137e-02,
          -1.2418e-01, -8.2354e-02, -6.5925e-02,  1.6098e-01,  2.0464e-01,
          -1.9857e-01, -2.0398e-01,  1.9966e-01,  7.9598e-02,  9.1483e-02,
          -1.9540e-01, -1.8827e-01,  1.1800e-01,  1.0289e-02, -1.6652e-01,
          -1.4606e-01,  1.5645e-01, -1.4856e-02, -7.9164e-02,  7.6735e-02,
           1.7878e-01, -7.5069e-02,  5.1754e-02],
         [-9.4692e-02, -9.0438e-03, -1.7975e-01,  1.9308e-01, -1.1868e-01,
           9.3873e-02,  1.7242e-01,  5.1886e-02, -1.6992e-01,  2.0724e-01,
          -1.5109e-01, -2.0970e-01,  2.0001e-01, -9.3307e-02,  7.9012e-02,
           2.8222e-02,  7.7008e-02, -4.8786e-02, -5.6962e-02,  1.2592e-01,
          -8.4403e-02, -2.0853e-01, -1.2928e-01,  7.0149e-02,  1.9941e-01,
          -8.3460e-02, -4.3382e-02,  1.0275e-01,  1.3896e-01,  3.3927e-04,
           1.4745e-01,  2.2443e-02, -1.9864e-01, -1.3659e-01,  1.7308e-02,
           2.9417e-02,  1.7372e-01,  1.2156e-01, -2.0657e-01,  6.2092e-02,
          -1.9462e-01, -2.0188e-01,  9.7465e-02, -5.2699e-02,  2.0690e-01,
          -1.7286e-01, -9.1221e-02,  1.8100e-01, -1.1266e-02,  1.4269e-01,
          -8.4888e-02,  5.7984e-02,  1.0336e-01,  1.0414e-01,  5.7299e-02,
          -1.6422e-01, -1.8251e-01,  4.0851e-02, -8.4911e-02, -2.0497e-01,
          -2.2152e-02,  2.0658e-01,  6.5733e-03,  1.9906e-01,  1.9123e-01,
          -4.5345e-02,  4.0588e-02,  2.0249e-01,  1.8504e-01, -4.6624e-02,
          -1.6234e-01,  1.0953e-01,  7.0963e-02, -1.7075e-02, -2.2418e-02,
          -6.1336e-02,  2.5505e-02, -3.4132e-02, -2.0396e-01,  2.0107e-01,
           1.4980e-01, -7.9274e-02, -2.0707e-01,  9.4527e-02, -1.2853e-01,
          -1.1666e-01, -1.4583e-01,  5.6859e-02, -1.8097e-01, -1.3803e-03,
           1.4468e-01, -1.4511e-01, -3.1703e-02,  1.2184e-01,  1.8216e-01,
           1.0928e-02, -5.6764e-02,  1.0388e-01,  4.5658e-02, -9.5442e-02,
           1.1397e-02,  2.1030e-01, -1.8186e-01, -1.0676e-02, -1.7776e-01,
           9.4723e-02, -3.1116e-02,  1.5478e-01, -1.1910e-01, -6.5264e-02,
          -1.8053e-01, -1.6024e-01,  9.0541e-02, -1.5165e-01, -1.9372e-01,
           1.1056e-03,  5.9977e-02, -2.6878e-02,  1.3146e-01, -1.0434e-01,
          -1.0082e-02,  1.5431e-01, -2.2912e-02, -1.3517e-01,  1.3392e-01,
          -7.1336e-02, -2.0140e-01,  1.2860e-01]]], requires_grad=True)
bias: Parameter containing:
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=True)
concat: True
dropout: 0.0
dump_patches: False
flow: 'source_to_target'
fuse: False
heads: 4
in_channels: (-1, -1)
inspector: <torch_geometric.nn.conv.utils.inspector.Inspector object at 0x7f690e103ed0>
lin_dst: Linear(-1, 512, bias=False)
lin_src: Linear(-1, 512, bias=False)
negative_slope: 0.2
node_dim: 0
out_channels: 128
special_args: {'size_i', 'size_j', 'index', 'dim_size', 'edge_index_i', 'ptr', 'adj_t', 'edge_index', 'size', 'edge_index_j'}
training: True
_aggregate_forward_hooks: OrderedDict([])
_aggregate_forward_pre_hooks: OrderedDict([])
_alpha: None
_apply: <bound method Module._apply of GATConv((-1, -1), 128, heads=4)>
_backward_hooks: OrderedDict([])
_buffers: OrderedDict([])
_call_impl: <bound method Module._call_impl of GATConv((-1, -1), 128, heads=4)>
_forward_hooks: OrderedDict([])
_forward_pre_hooks: OrderedDict([])
_get_backward_hooks: <bound method Module._get_backward_hooks of GATConv((-1, -1), 128, heads=4)>
_get_name: <bound method Module._get_name of GATConv((-1, -1), 128, heads=4)>
_is_full_backward_hook: None
_load_from_state_dict: <bound method Module._load_from_state_dict of GATConv((-1, -1), 128, heads=4)>
_load_state_dict_pre_hooks: OrderedDict([])
_maybe_warn_non_full_backward_hook: <bound method Module._maybe_warn_non_full_backward_hook of GATConv((-1, -1), 128, heads=4)>
_message_and_aggregate_forward_hooks: OrderedDict([])
_message_and_aggregate_forward_pre_hooks: OrderedDict([])
_message_forward_hooks: OrderedDict([])
_message_forward_pre_hooks: OrderedDict([])
_modules: OrderedDict([('lin_src', Linear(-1, 512, bias=False)), ('lin_dst', Linear(-1, 512, bias=False))])
_named_members: <bound method Module._named_members of GATConv((-1, -1), 128, heads=4)>
_non_persistent_buffers_set: {}
_parameters: OrderedDict([('att_src', Parameter containing...grad=True)), ('att_dst', Parameter containing...grad=True)), ('bias', Parameter containing...grad=True))])
_propagate_forward_hooks: OrderedDict([])
_propagate_forward_pre_hooks: OrderedDict([])
_register_load_state_dict_pre_hook: <bound method Module._register_load_state_dict_pre_hook of GATConv((-1, -1), 128, heads=4)>
_register_state_dict_hook: <bound method Module._register_state_dict_hook of GATConv((-1, -1), 128, heads=4)>
_replicate_for_data_parallel: <bound method Module._replicate_for_data_parallel of GATConv((-1, -1), 128, heads=4)>
_save_to_state_dict: <bound method Module._save_to_state_dict of GATConv((-1, -1), 128, heads=4)>
_slow_forward: <bound method Module._slow_forward of GATConv((-1, -1), 128, heads=4)>
_state_dict_hooks: OrderedDict([])
_version: 1
len(): 1
_named_members: <bound method Module._named_members of ModuleList(
  (0): GATConv((-1, -1), 128, heads=4)
)>
_non_persistent_buffers_set: {}
_parameters: OrderedDict([])
_register_load_state_dict_pre_hook: <bound method Module._register_load_state_dict_pre_hook of ModuleList(
  (0): GATConv((-1, -1), 128, heads=4)
)>
_register_state_dict_hook: <bound method Module._register_state_dict_hook of ModuleList(
  (0): GATConv((-1, -1), 128, heads=4)
)>
_replicate_for_data_parallel: <bound method Module._replicate_for_data_parallel of ModuleList(
  (0): GATConv((-1, -1), 128, heads=4)
)>
_save_to_state_dict: <bound method Module._save_to_state_dict of ModuleList(
  (0): GATConv((-1, -1), 128, heads=4)
)>
_slow_forward: <bound method Module._slow_forward of ModuleList(
  (0): GATConv((-1, -1), 128, heads=4)
)>
_state_dict_hooks: OrderedDict([])
_version: 1
dump_patches: False
heads: 4
hidden_dim: 128
lins: ModuleList()
num_classes: 349
num_layers: 2
training: True
_apply: <bound method Module._apply of GAT(
  (convs): ModuleList(
    (0): GATConv((-1, -1), 128, heads=4)
  )
  (bns): ModuleList()
  (lins): ModuleList()
)>
_backward_hooks: OrderedDict([])
_buffers: OrderedDict([])
_call_impl: <bound method Module._call_impl of GAT(
  (convs): ModuleList(
    (0): GATConv((-1, -1), 128, heads=4)
  )
  (bns): ModuleList()
  (lins): ModuleList()
)>
_forward_hooks: OrderedDict([])
_forward_pre_hooks: OrderedDict([])
_get_backward_hooks: <bound method Module._get_backward_hooks of GAT(
  (convs): ModuleList(
    (0): GATConv((-1, -1), 128, heads=4)
  )
  (bns): ModuleList()
  (lins): ModuleList()
)>
_get_name: <bound method Module._get_name of GAT(
  (convs): ModuleList(
    (0): GATConv((-1, -1), 128, heads=4)
  )
  (bns): ModuleList()
  (lins): ModuleList()
)>
_is_full_backward_hook: None
_load_from_state_dict: <bound method Module._load_from_state_dict of GAT(
  (convs): ModuleList(
    (0): GATConv((-1, -1), 128, heads=4)
  )
  (bns): ModuleList()
  (lins): ModuleList()
)>
_load_state_dict_pre_hooks: OrderedDict([])
_maybe_warn_non_full_backward_hook: <bound method Module._maybe_warn_non_full_backward_hook of GAT(
  (convs): ModuleList(
    (0): GATConv((-1, -1), 128, heads=4)
  )
  (bns): ModuleList()
  (lins): ModuleList()
)>
_modules: OrderedDict([('convs', ModuleList(
  (0): G...heads=4)
)), ('bns', ModuleList()), ('lins', ModuleList())])
_named_members: <bound method Module._named_members of GAT(
  (convs): ModuleList(
    (0): GATConv((-1, -1), 128, heads=4)
  )
  (bns): ModuleList()
  (lins): ModuleList()
)>
_non_persistent_buffers_set: {}
_parameters: OrderedDict([])
_register_load_state_dict_pre_hook: <bound method Module._register_load_state_dict_pre_hook of GAT(
  (convs): ModuleList(
    (0): GATConv((-1, -1), 128, heads=4)
  )
  (bns): ModuleList()
  (lins): ModuleList()
)>
_register_state_dict_hook: <bound method Module._register_state_dict_hook of GAT(
  (convs): ModuleList(
    (0): GATConv((-1, -1), 128, heads=4)
  )
  (bns): ModuleList()
  (lins): ModuleList()
)>
_replicate_for_data_parallel: <bound method Module._replicate_for_data_parallel of GAT(
  (convs): ModuleList(
    (0): GATConv((-1, -1), 128, heads=4)
  )
  (bns): ModuleList()
  (lins): ModuleList()
)>
_save_to_state_dict: <bound method Module._save_to_state_dict of GAT(
  (convs): ModuleList(
    (0): GATConv((-1, -1), 128, heads=4)
  )
  (bns): ModuleList()
  (lins): ModuleList()
)>
_slow_forward: <bound method Module._slow_forward of GAT(
  (convs): ModuleList(
    (0): GATConv((-1, -1), 128, heads=4)
  )
  (bns): ModuleList()
  (lins): ModuleList()
)>
_state_dict_hooks: OrderedDict([])
_version: 1
sel
Traceback (most recent call last):
  File "<string>", line 1, in <module>
NameError: name 'sel' is not defined
self.lins
ModuleList(
  (0): Linear(-1, 512, bias=True)
)
special variables:
function variables:
T_destination: ~T_destination
dump_patches: False
training: True
_apply: <bound method Module._apply of ModuleList(
  (0): Linear(-1, 512, bias=True)
)>
_backward_hooks: OrderedDict([])
_buffers: OrderedDict([])
_call_impl: <bound method Module._call_impl of ModuleList(
  (0): Linear(-1, 512, bias=True)
)>
_forward_hooks: OrderedDict([])
_forward_pre_hooks: OrderedDict([])
_get_abs_string_index: <bound method ModuleList._get_abs_string_index of ModuleList(
  (0): Linear(-1, 512, bias=True)
)>
_get_backward_hooks: <bound method Module._get_backward_hooks of ModuleList(
  (0): Linear(-1, 512, bias=True)
)>
_get_name: <bound method Module._get_name of ModuleList(
  (0): Linear(-1, 512, bias=True)
)>
_is_full_backward_hook: None
_load_from_state_dict: <bound method Module._load_from_state_dict of ModuleList(
  (0): Linear(-1, 512, bias=True)
)>
_load_state_dict_pre_hooks: OrderedDict([])
_maybe_warn_non_full_backward_hook: <bound method Module._maybe_warn_non_full_backward_hook of ModuleList(
  (0): Linear(-1, 512, bias=True)
)>
_modules: OrderedDict([('0', Linear(-1, 512, bias=True))])
special variables:
function variables:
'0': Linear(-1, 512, bias=True)
len(): 1
_named_members: <bound method Module._named_members of ModuleList(
  (0): Linear(-1, 512, bias=True)
)>
_non_persistent_buffers_set: {}
_parameters: OrderedDict([])
_register_load_state_dict_pre_hook: <bound method Module._register_load_state_dict_pre_hook of ModuleList(
  (0): Linear(-1, 512, bias=True)
)>
_register_state_dict_hook: <bound method Module._register_state_dict_hook of ModuleList(
  (0): Linear(-1, 512, bias=True)
)>
_replicate_for_data_parallel: <bound method Module._replicate_for_data_parallel of ModuleList(
  (0): Linear(-1, 512, bias=True)
)>
_save_to_state_dict: <bound method Module._save_to_state_dict of ModuleList(
  (0): Linear(-1, 512, bias=True)
)>
_slow_forward: <bound method Module._slow_forward of ModuleList(
  (0): Linear(-1, 512, bias=True)
)>
_state_dict_hooks: OrderedDict([])
_version: 1
self
<torch_geometric.nn.to_hetero_transformer.ToHeteroTransformer object at 0x7f690e103a10>
self
<torch_geometric.nn.to_hetero_transformer.ToHeteroTransformer object at 0x7f690e103a10>
special variables:
function variables:
aggr: 'sum'
aggrs: {'sum': <built-in method add...a14282c60>, 'mean': <built-in method add...a14282c60>, 'max': <built-in method max...a14282c60>, 'min': <built-in method min...a14282c60>, 'mul': <built-in method mul...a14282c60>}
debug: False
gm: GraphModule(
  (convs): Module(
    (0): GATConv((-1, -1), 128, heads=4)
    (1): GATConv((-1, -1), 128, heads=4)
  )
  (lins): Module(
    (0): Linear(-1, 512, bias=True)
    (1): Linear(-1, 512, bias=True)
  )
  (bns): Module(
    (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc_out): Linear(-1, 349, bias=True)
)
special variables:
function variables:
T_destination: ~T_destination
bns: Module(
  (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
code: 'import torch\ndef forward(self, x, edge_index):\n    convs_0 = getattr(self.convs, "0")(x, edge_index)\n    lins_0 = getattr(self.lins, "0")(x);  x = None\n    add_1 = convs_0 + lins_0;  convs_0 = lins_0 = None\n    bns_0 = getattr(self.bns, "0")(add_1);  add_1 = None\n    relu_1 = torch.nn.functional.relu(bns_0, inplace = True);  bns_0 = None\n    dropout_1 = torch.nn.functional.dropout(relu_1, p = 0.5, training = True, inplace = False);  relu_1 = None\n    convs_1 = getattr(self.convs, "1")(dropout_1, edge_index);  edge_index = None\n    lins_1 = getattr(self.lins, "1")(dropout_1);  dropout_1 = None\n    add_2 = convs_1 + lins_1;  convs_1 = lins_1 = None\n    bns_1 = getattr(self.bns, "1")(add_2);  add_2 = None\n    relu_2 = torch.nn.functional.relu(bns_1, inplace = True);  bns_1 = None\n    dropout_2 = torch.nn.functional.dropout(relu_2, p = 0.5, training = True, inplace = False);  relu_2 = None\n    fc_out = self.fc_out(dropout_2);  dropout_2 = None\n    return fc_out\n    '
convs: Module(
  (0): GATConv((-1, -1), 128, heads=4)
  (1): GATConv((-1, -1), 128, heads=4)
)
dump_patches: False
fc_out: Linear(-1, 349, bias=True)
graph: <torch.fx.graph.Graph object at 0x7f690e09bcd0>
lins: Module(
  (0): Linear(-1, 512, bias=True)
  (1): Linear(-1, 512, bias=True)
)
training: True
_apply: <bound method Module._apply of GraphModule(
  (convs): Module(
    (0): GATConv((-1, -1), 128, heads=4)
    (1): GATConv((-1, -1), 128, heads=4)
  )
  (lins): Module(
    (0): Linear(-1, 512, bias=True)
    (1): Linear(-1, 512, bias=True)
  )
  (bns): Module(
    (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc_out): Linear(-1, 349, bias=True)
)>
_backward_hooks: OrderedDict([])
_buffers: OrderedDict([])
_call_impl: <bound method Module._call_impl of GraphModule(
  (convs): Module(
    (0): GATConv((-1, -1), 128, heads=4)
    (1): GATConv((-1, -1), 128, heads=4)
  )
  (lins): Module(
    (0): Linear(-1, 512, bias=True)
    (1): Linear(-1, 512, bias=True)
  )
  (bns): Module(
    (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc_out): Linear(-1, 349, bias=True)
)>
_code: 'import torch\ndef forward(self, x, edge_index):\n    convs_0 = getattr(self.convs, "0")(x, edge_index)\n    lins_0 = getattr(self.lins, "0")(x);  x = None\n    add_1 = convs_0 + lins_0;  convs_0 = lins_0 = None\n    bns_0 = getattr(self.bns, "0")(add_1);  add_1 = None\n    relu_1 = torch.nn.functional.relu(bns_0, inplace = True);  bns_0 = None\n    dropout_1 = torch.nn.functional.dropout(relu_1, p = 0.5, training = True, inplace = False);  relu_1 = None\n    convs_1 = getattr(self.convs, "1")(dropout_1, edge_index);  edge_index = None\n    lins_1 = getattr(self.lins, "1")(dropout_1);  dropout_1 = None\n    add_2 = convs_1 + lins_1;  convs_1 = lins_1 = None\n    bns_1 = getattr(self.bns, "1")(add_2);  add_2 = None\n    relu_2 = torch.nn.functional.relu(bns_1, inplace = True);  bns_1 = None\n    dropout_2 = torch.nn.functional.dropout(relu_2, p = 0.5, training = True, inplace = False);  relu_2 = None\n    fc_out = self.fc_out(dropout_2);  dropout_2 = None\n    return fc_out\n    '
_forward_hooks: OrderedDict([])
_forward_pre_hooks: OrderedDict([])
_get_backward_hooks: <bound method Module._get_backward_hooks of GraphModule(
  (convs): Module(
    (0): GATConv((-1, -1), 128, heads=4)
    (1): GATConv((-1, -1), 128, heads=4)
  )
  (lins): Module(
    (0): Linear(-1, 512, bias=True)
    (1): Linear(-1, 512, bias=True)
  )
  (bns): Module(
    (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc_out): Linear(-1, 349, bias=True)
)>
_get_name: <bound method Module._get_name of GraphModule(
  (convs): Module(
    (0): GATConv((-1, -1), 128, heads=4)
    (1): GATConv((-1, -1), 128, heads=4)
  )
  (lins): Module(
    (0): Linear(-1, 512, bias=True)
    (1): Linear(-1, 512, bias=True)
  )
  (bns): Module(
    (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc_out): Linear(-1, 349, bias=True)
)>
_graph: <torch.fx.graph.Graph object at 0x7f690e09bcd0>
_is_full_backward_hook: None
_load_from_state_dict: <bound method Module._load_from_state_dict of GraphModule(
  (convs): Module(
    (0): GATConv((-1, -1), 128, heads=4)
    (1): GATConv((-1, -1), 128, heads=4)
  )
  (lins): Module(
    (0): Linear(-1, 512, bias=True)
    (1): Linear(-1, 512, bias=True)
  )
  (bns): Module(
    (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc_out): Linear(-1, 349, bias=True)
)>
_load_state_dict_pre_hooks: OrderedDict([])
_maybe_warn_non_full_backward_hook: <bound method Module._maybe_warn_non_full_backward_hook of GraphModule(
  (convs): Module(
    (0): GATConv((-1, -1), 128, heads=4)
    (1): GATConv((-1, -1), 128, heads=4)
  )
  (lins): Module(
    (0): Linear(-1, 512, bias=True)
    (1): Linear(-1, 512, bias=True)
  )
  (bns): Module(
    (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc_out): Linear(-1, 349, bias=True)
)>
_modules: OrderedDict([('convs', Module(
  (0): GATCo...heads=4)
)), ('lins', Module(
  (0): Linea...as=True)
)), ('bns', Module(
  (0): Batch...ts=True)
)), ('fc_out', Linear(-1, 349, bias=True))])
_named_members: <bound method Module._named_members of GraphModule(
  (convs): Module(
    (0): GATConv((-1, -1), 128, heads=4)
    (1): GATConv((-1, -1), 128, heads=4)
  )
  (lins): Module(
    (0): Linear(-1, 512, bias=True)
    (1): Linear(-1, 512, bias=True)
  )
  (bns): Module(
    (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc_out): Linear(-1, 349, bias=True)
)>
_non_persistent_buffers_set: {}
_parameters: OrderedDict([])
_register_load_state_dict_pre_hook: <bound method Module._register_load_state_dict_pre_hook of GraphModule(
  (convs): Module(
    (0): GATConv((-1, -1), 128, heads=4)
    (1): GATConv((-1, -1), 128, heads=4)
  )
  (lins): Module(
    (0): Linear(-1, 512, bias=True)
    (1): Linear(-1, 512, bias=True)
  )
  (bns): Module(
    (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc_out): Linear(-1, 349, bias=True)
)>
_register_state_dict_hook: <bound method Module._register_state_dict_hook of GraphModule(
  (convs): Module(
    (0): GATConv((-1, -1), 128, heads=4)
    (1): GATConv((-1, -1), 128, heads=4)
  )
  (lins): Module(
    (0): Linear(-1, 512, bias=True)
    (1): Linear(-1, 512, bias=True)
  )
  (bns): Module(
    (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc_out): Linear(-1, 349, bias=True)
)>
_replicate_for_data_parallel: <bound method Module._replicate_for_data_parallel of GraphModule(
  (convs): Module(
    (0): GATConv((-1, -1), 128, heads=4)
    (1): GATConv((-1, -1), 128, heads=4)
  )
  (lins): Module(
    (0): Linear(-1, 512, bias=True)
    (1): Linear(-1, 512, bias=True)
  )
  (bns): Module(
    (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc_out): Linear(-1, 349, bias=True)
)>
_save_to_state_dict: <bound method Module._save_to_state_dict of GraphModule(
  (convs): Module(
    (0): GATConv((-1, -1), 128, heads=4)
    (1): GATConv((-1, -1), 128, heads=4)
  )
  (lins): Module(
    (0): Linear(-1, 512, bias=True)
    (1): Linear(-1, 512, bias=True)
  )
  (bns): Module(
    (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc_out): Linear(-1, 349, bias=True)
)>
_slow_forward: <bound method Module._slow_forward of GraphModule(
  (convs): Module(
    (0): GATConv((-1, -1), 128, heads=4)
    (1): GATConv((-1, -1), 128, heads=4)
  )
  (lins): Module(
    (0): Linear(-1, 512, bias=True)
    (1): Linear(-1, 512, bias=True)
  )
  (bns): Module(
    (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc_out): Linear(-1, 349, bias=True)
)>
_state_dict_hooks: OrderedDict([])
_version: 1
graph: <torch.fx.graph.Graph object at 0x7f690e09bcd0>
input_map: None
metadata: (['paper', 'author', 'institution', 'field_of_study'], [(...), (...), (...), (...), (...), (...), (...)])
module: GAT(
  (convs): ModuleList(
    (0): GATConv((-1, -1), 128, heads=4)
    (1): GATConv((-1, -1), 128, heads=4)
  )
  (bns): ModuleList(
    (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (lins): ModuleList(
    (0): Linear(-1, 512, bias=True)
    (1): Linear(-1, 512, bias=True)
  )
  (fc_out): Linear(-1, 349, bias=True)
)
_init_submodule: <bound method Transformer._init_submodule of <torch_geometric.nn.to_hetero_transformer.ToHeteroTransformer object at 0x7f690e103a10>>
_state: {}
model
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
special variables:
function variables:
T_destination: ~T_destination
bns: ModuleList(
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
code: 'import torch\ndef forward(self, x, edge_index):\n    x__paper = x.get(\'paper\')\n    x__author = x.get(\'author\')\n    x__institution = x.get(\'institution\')\n    x__field_of_study = x.get(\'field_of_study\');  x = None\n    edge_index__author__affiliated_with__institution = edge_index.get((\'author\', \'affiliated_with\', \'institution\'))\n    edge_index__author__writes__paper = edge_index.get((\'author\', \'writes\', \'paper\'))\n    edge_index__paper__cites__paper = edge_index.get((\'paper\', \'cites\', \'paper\'))\n    edge_index__paper__has_topic__field_of_study = edge_index.get((\'paper\', \'has_topic\', \'field_of_study\'))\n    edge_index__institution__rev_affiliated_with__author = edge_index.get((\'institution\', \'rev_affiliated_with\', \'author\'))\n    edge_index__paper__rev_writes__author = edge_index.get((\'paper\', \'rev_writes\', \'author\'))\n    edge_index__field_of_study__rev_has_topic__paper = edge_index.get((\'field_of_study\', \'rev_has_topic\', \'paper\'));  edge_index = None\n    convs_0__institution = getattr(self.convs, "0").author__affiliated_with__institution((x__author, x__institution), edge_index__author__affiliated_with__institution)\n    convs_0__paper1 = getattr(self.convs, "0").author__writes__paper((x__author, x__paper), edge_index__author__writes__paper)\n    convs_0__paper2 = getattr(self.convs, "0").paper__cites__paper((x__paper, x__paper), edge_index__paper__cites__paper)\n    convs_0__field_of_study = getattr(self.convs, "0").paper__has_topic__field_of_study((x__paper, x__field_of_study), edge_index__paper__has_topic__field_of_study)\n    convs_0__author1 = getattr(self.convs, "0").institution__rev_affiliated_with__author((x__institution, x__author), edge_index__institution__rev_affiliated_with__author)\n    convs_0__author2 = getattr(self.convs, "0").paper__rev_writes__author((x__paper, x__author), edge_index__paper__rev_writes__author)\n    convs_0__paper3 = getattr(self.convs, "0").field_of_study__rev_has_topic__paper((x__field_of_study, x__paper), edge_index__field_of_study__rev_has_topic__paper)\n    convs_0__paper4 = torch.add(convs_0__paper1, convs_0__paper2);  convs_0__paper1 = convs_0__paper2 = None\n    convs_0__paper = torch.add(convs_0__paper3, convs_0__paper4);  convs_0__paper3 = convs_0__paper4 = None\n    convs_0__author = torch.add(convs_0__author1, convs_0__author2);  convs_0__author1 = convs_0__author2 = None\n    lins_0__paper = getattr(self.lins, "0").paper(x__paper);  x__paper = None\n    lins_0__author = getattr(self.lins, "0").author(x__author);  x__author = None\n    lins_0__institution = getattr(self.lins, "0").institution(x__institution);  x__institution = None\n    lins_0__field_of_study = getattr(self.lins, "0").field_of_study(x__field_of_study);  x__field_of_study = None\n    add_1__paper = convs_0__paper + lins_0__paper;  convs_0__paper = lins_0__paper = None\n    add_1__author = convs_0__author + lins_0__author;  convs_0__author = lins_0__author = None\n    add_1__institution = convs_0__institution + lins_0__institution;  convs_0__institution = lins_0__institution = None\n    add_1__field_of_study = convs_0__field_of_study + lins_0__field_of_study;  convs_0__field_of_study = lins_0__field_of_study = None\n    bns_0__paper = getattr(self.bns, "0").paper(add_1__paper);  add_1__paper = None\n    bns_0__author = getattr(self.bns, "0").author(add_1__author);  add_1__author = None\n    bns_0__institution = getattr(self.bns, "0").institution(add_1__institution);  add_1__institution = None\n    bns_0__field_of_study = getattr(self.bns, "0").field_of_study(add_1__field_of_study);  add_1__field_of_study = None\n    relu_1__paper = torch.nn.functional.relu(bns_0__paper, inplace = True);  bns_0__paper = None\n    relu_1__author = torch.nn.functional.relu(bns_0__author, inplace = True);  bns_0__author = None\n    relu_1__institution = torch.nn.functional.relu(bns_0__institution, inplace = True);  bns_0__institution = None\n    relu_1__field_of_study = torch.nn.functional.relu(bns_0__field_of_study, inplace = True);  bns_0__field_of_study = None\n    dropout_1__paper = torch.nn.functional.dropout(relu_1__paper, p = 0.5, training = True, inplace = False);  relu_1__paper = None\n    dropout_1__author = torch.nn.functional.dropout(relu_1__author, p = 0.5, training = True, inplace = False);  relu_1__author = None\n    dropout_1__institution = torch.nn.functional.dropout(relu_1__institution, p = 0.5, training = True, inplace = False);  relu_1__institution = None\n    dropout_1__field_of_study = torch.nn.functional.dropout(relu_1__field_of_study, p = 0.5, training = True, inplace = False);  relu_1__field_of_study = None\n    convs_1__institution = getattr(self.convs, "1").author__affiliated_with__institution((dropout_1__author, dropout_1__institution), edge_index__author__affiliated_with__institution);  edge_index__author__affiliated_with__institution = None\n    convs_1__paper1 = getattr(self.convs, "1").author__writes__paper((dropout_1__author, dropout_1__paper), edge_index__author__writes__paper);  edge_index__author__writes__paper = None\n    convs_1__paper2 = getattr(self.convs, "1").paper__cites__paper((dropout_1__paper, dropout_1__paper), edge_index__paper__cites__paper);  edge_index__paper__cites__paper = None\n    convs_1__field_of_study = getattr(self.convs, "1").paper__has_topic__field_of_study((dropout_1__paper, dropout_1__field_of_study), edge_index__paper__has_topic__field_of_study);  edge_index__paper__has_topic__field_of_study = None\n    convs_1__author1 = getattr(self.convs, "1").institution__rev_affiliated_with__author((dropout_1__institution, dropout_1__author), edge_index__institution__rev_affiliated_with__author);  edge_index__institution__rev_affiliated_with__author = None\n    convs_1__author2 = getattr(self.convs, "1").paper__rev_writes__author((dropout_1__paper, dropout_1__author), edge_index__paper__rev_writes__author);  edge_index__paper__rev_writes__author = None\n    convs_1__paper3 = getattr(self.convs, "1").field_of_study__rev_has_topic__paper((dropout_1__field_of_study, dropout_1__paper), edge_index__field_of_study__rev_has_topic__paper);  edge_index__field_of_study__rev_has_topic__paper = None\n    convs_1__paper4 = torch.add(convs_1__paper1, convs_1__paper2);  convs_1__paper1 = convs_1__paper2 = None\n    convs_1__paper = torch.add(convs_1__paper3, convs_1__paper4);  convs_1__paper3 = convs_1__paper4 = None\n    convs_1__author = torch.add(convs_1__author1, convs_1__author2);  convs_1__author1 = convs_1__author2 = None\n    lins_1__paper = getattr(self.lins, "1").paper(dropout_1__paper);  dropout_1__paper = None\n    lins_1__author = getattr(self.lins, "1").author(dropout_1__author);  dropout_1__author = None\n    lins_1__institution = getattr(self.lins, "1").institution(dropout_1__institution);  dropout_1__institution = None\n    lins_1__field_of_study = getattr(self.lins, "1").field_of_study(dropout_1__field_of_study);  dropout_1__field_of_study = None\n    add_2__paper = convs_1__paper + lins_1__paper;  convs_1__paper = lins_1__paper = None\n    add_2__author = convs_1__author + lins_1__author;  convs_1__author = lins_1__author = None\n    add_2__institution = convs_1__institution + lins_1__institution;  convs_1__institution = lins_1__institution = None\n    add_2__field_of_study = convs_1__field_of_study + lins_1__field_of_study;  convs_1__field_of_study = lins_1__field_of_study = None\n    bns_1__paper = getattr(self.bns, "1").paper(add_2__paper);  add_2__paper = None\n    bns_1__author = getattr(self.bns, "1").author(add_2__author);  add_2__author = None\n    bns_1__institution = getattr(self.bns, "1").institution(add_2__institution);  add_2__institution = None\n    bns_1__field_of_study = getattr(self.bns, "1").field_of_study(add_2__field_of_study);  add_2__field_of_study = None\n    relu_2__paper = torch.nn.functional.relu(bns_1__paper, inplace = True);  bns_1__paper = None\n    relu_2__author = torch.nn.functional.relu(bns_1__author, inplace = True);  bns_1__author = None\n    relu_2__institution = torch.nn.functional.relu(bns_1__institution, inplace = True);  bns_1__institution = None\n    relu_2__field_of_study = torch.nn.functional.relu(bns_1__field_of_study, inplace = True);  bns_1__field_of_study = None\n    dropout_2__paper = torch.nn.functional.dropout(relu_2__paper, p = 0.5, training = True, inplace = False);  relu_2__paper = None\n    dropout_2__author = torch.nn.functional.dropout(relu_2__author, p = 0.5, training = True, inplace = False);  relu_2__author = None\n    dropout_2__institution = torch.nn.functional.dropout(relu_2__institution, p = 0.5, training = True, inplace = False);  relu_2__institution = None\n    dropout_2__field_of_study = torch.nn.functional.dropout(relu_2__field_of_study, p = 0.5, training = True, inplace = False);  relu_2__field_of_study = None\n    fc_out__paper = self.fc_out.paper(dropout_2__paper);  dropout_2__paper = None\n    fc_out__author = self.fc_out.author(dropout_2__author);  dropout_2__author = None\n    fc_out__institution = self.fc_out.institution(dropout_2__institution);  dropout_2__institution = None\n    fc_out__field_of_study = self.fc_out.field_of_study(dropout_2__field_of_study);  dropout_2__field_of_study = None\n    return {\'paper\': fc_out__paper, \'author\': fc_out__author, \'institution\': fc_out__institution, \'field_of_study\': fc_out__field_of_study}\n    '
convs: ModuleList(
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
dump_patches: False
fc_out: ModuleDict(
  (paper): Linear(-1, 349, bias=True)
  (author): Linear(-1, 349, bias=True)
  (institution): Linear(-1, 349, bias=True)
  (field_of_study): Linear(-1, 349, bias=True)
)
graph: <torch.fx.graph.Graph object at 0x7f690e09bcd0>
lins: ModuleList(
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
training: True
_apply: <bound method Module._apply of GraphModule(
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
)>
_backward_hooks: OrderedDict([])
_buffers: OrderedDict([])
_call_impl: <bound method Module._call_impl of GraphModule(
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
)>
_code: 'import torch\ndef forward(self, x, edge_index):\n    x__paper = x.get(\'paper\')\n    x__author = x.get(\'author\')\n    x__institution = x.get(\'institution\')\n    x__field_of_study = x.get(\'field_of_study\');  x = None\n    edge_index__author__affiliated_with__institution = edge_index.get((\'author\', \'affiliated_with\', \'institution\'))\n    edge_index__author__writes__paper = edge_index.get((\'author\', \'writes\', \'paper\'))\n    edge_index__paper__cites__paper = edge_index.get((\'paper\', \'cites\', \'paper\'))\n    edge_index__paper__has_topic__field_of_study = edge_index.get((\'paper\', \'has_topic\', \'field_of_study\'))\n    edge_index__institution__rev_affiliated_with__author = edge_index.get((\'institution\', \'rev_affiliated_with\', \'author\'))\n    edge_index__paper__rev_writes__author = edge_index.get((\'paper\', \'rev_writes\', \'author\'))\n    edge_index__field_of_study__rev_has_topic__paper = edge_index.get((\'field_of_study\', \'rev_has_topic\', \'paper\'));  edge_index = None\n    convs_0__institution = getattr(self.convs, "0").author__affiliated_with__institution((x__author, x__institution), edge_index__author__affiliated_with__institution)\n    convs_0__paper1 = getattr(self.convs, "0").author__writes__paper((x__author, x__paper), edge_index__author__writes__paper)\n    convs_0__paper2 = getattr(self.convs, "0").paper__cites__paper((x__paper, x__paper), edge_index__paper__cites__paper)\n    convs_0__field_of_study = getattr(self.convs, "0").paper__has_topic__field_of_study((x__paper, x__field_of_study), edge_index__paper__has_topic__field_of_study)\n    convs_0__author1 = getattr(self.convs, "0").institution__rev_affiliated_with__author((x__institution, x__author), edge_index__institution__rev_affiliated_with__author)\n    convs_0__author2 = getattr(self.convs, "0").paper__rev_writes__author((x__paper, x__author), edge_index__paper__rev_writes__author)\n    convs_0__paper3 = getattr(self.convs, "0").field_of_study__rev_has_topic__paper((x__field_of_study, x__paper), edge_index__field_of_study__rev_has_topic__paper)\n    convs_0__paper4 = torch.add(convs_0__paper1, convs_0__paper2);  convs_0__paper1 = convs_0__paper2 = None\n    convs_0__paper = torch.add(convs_0__paper3, convs_0__paper4);  convs_0__paper3 = convs_0__paper4 = None\n    convs_0__author = torch.add(convs_0__author1, convs_0__author2);  convs_0__author1 = convs_0__author2 = None\n    lins_0__paper = getattr(self.lins, "0").paper(x__paper);  x__paper = None\n    lins_0__author = getattr(self.lins, "0").author(x__author);  x__author = None\n    lins_0__institution = getattr(self.lins, "0").institution(x__institution);  x__institution = None\n    lins_0__field_of_study = getattr(self.lins, "0").field_of_study(x__field_of_study);  x__field_of_study = None\n    add_1__paper = convs_0__paper + lins_0__paper;  convs_0__paper = lins_0__paper = None\n    add_1__author = convs_0__author + lins_0__author;  convs_0__author = lins_0__author = None\n    add_1__institution = convs_0__institution + lins_0__institution;  convs_0__institution = lins_0__institution = None\n    add_1__field_of_study = convs_0__field_of_study + lins_0__field_of_study;  convs_0__field_of_study = lins_0__field_of_study = None\n    bns_0__paper = getattr(self.bns, "0").paper(add_1__paper);  add_1__paper = None\n    bns_0__author = getattr(self.bns, "0").author(add_1__author);  add_1__author = None\n    bns_0__institution = getattr(self.bns, "0").institution(add_1__institution);  add_1__institution = None\n    bns_0__field_of_study = getattr(self.bns, "0").field_of_study(add_1__field_of_study);  add_1__field_of_study = None\n    relu_1__paper = torch.nn.functional.relu(bns_0__paper, inplace = True);  bns_0__paper = None\n    relu_1__author = torch.nn.functional.relu(bns_0__author, inplace = True);  bns_0__author = None\n    relu_1__institution = torch.nn.functional.relu(bns_0__institution, inplace = True);  bns_0__institution = None\n    relu_1__field_of_study = torch.nn.functional.relu(bns_0__field_of_study, inplace = True);  bns_0__field_of_study = None\n    dropout_1__paper = torch.nn.functional.dropout(relu_1__paper, p = 0.5, training = True, inplace = False);  relu_1__paper = None\n    dropout_1__author = torch.nn.functional.dropout(relu_1__author, p = 0.5, training = True, inplace = False);  relu_1__author = None\n    dropout_1__institution = torch.nn.functional.dropout(relu_1__institution, p = 0.5, training = True, inplace = False);  relu_1__institution = None\n    dropout_1__field_of_study = torch.nn.functional.dropout(relu_1__field_of_study, p = 0.5, training = True, inplace = False);  relu_1__field_of_study = None\n    convs_1__institution = getattr(self.convs, "1").author__affiliated_with__institution((dropout_1__author, dropout_1__institution), edge_index__author__affiliated_with__institution);  edge_index__author__affiliated_with__institution = None\n    convs_1__paper1 = getattr(self.convs, "1").author__writes__paper((dropout_1__author, dropout_1__paper), edge_index__author__writes__paper);  edge_index__author__writes__paper = None\n    convs_1__paper2 = getattr(self.convs, "1").paper__cites__paper((dropout_1__paper, dropout_1__paper), edge_index__paper__cites__paper);  edge_index__paper__cites__paper = None\n    convs_1__field_of_study = getattr(self.convs, "1").paper__has_topic__field_of_study((dropout_1__paper, dropout_1__field_of_study), edge_index__paper__has_topic__field_of_study);  edge_index__paper__has_topic__field_of_study = None\n    convs_1__author1 = getattr(self.convs, "1").institution__rev_affiliated_with__author((dropout_1__institution, dropout_1__author), edge_index__institution__rev_affiliated_with__author);  edge_index__institution__rev_affiliated_with__author = None\n    convs_1__author2 = getattr(self.convs, "1").paper__rev_writes__author((dropout_1__paper, dropout_1__author), edge_index__paper__rev_writes__author);  edge_index__paper__rev_writes__author = None\n    convs_1__paper3 = getattr(self.convs, "1").field_of_study__rev_has_topic__paper((dropout_1__field_of_study, dropout_1__paper), edge_index__field_of_study__rev_has_topic__paper);  edge_index__field_of_study__rev_has_topic__paper = None\n    convs_1__paper4 = torch.add(convs_1__paper1, convs_1__paper2);  convs_1__paper1 = convs_1__paper2 = None\n    convs_1__paper = torch.add(convs_1__paper3, convs_1__paper4);  convs_1__paper3 = convs_1__paper4 = None\n    convs_1__author = torch.add(convs_1__author1, convs_1__author2);  convs_1__author1 = convs_1__author2 = None\n    lins_1__paper = getattr(self.lins, "1").paper(dropout_1__paper);  dropout_1__paper = None\n    lins_1__author = getattr(self.lins, "1").author(dropout_1__author);  dropout_1__author = None\n    lins_1__institution = getattr(self.lins, "1").institution(dropout_1__institution);  dropout_1__institution = None\n    lins_1__field_of_study = getattr(self.lins, "1").field_of_study(dropout_1__field_of_study);  dropout_1__field_of_study = None\n    add_2__paper = convs_1__paper + lins_1__paper;  convs_1__paper = lins_1__paper = None\n    add_2__author = convs_1__author + lins_1__author;  convs_1__author = lins_1__author = None\n    add_2__institution = convs_1__institution + lins_1__institution;  convs_1__institution = lins_1__institution = None\n    add_2__field_of_study = convs_1__field_of_study + lins_1__field_of_study;  convs_1__field_of_study = lins_1__field_of_study = None\n    bns_1__paper = getattr(self.bns, "1").paper(add_2__paper);  add_2__paper = None\n    bns_1__author = getattr(self.bns, "1").author(add_2__author);  add_2__author = None\n    bns_1__institution = getattr(self.bns, "1").institution(add_2__institution);  add_2__institution = None\n    bns_1__field_of_study = getattr(self.bns, "1").field_of_study(add_2__field_of_study);  add_2__field_of_study = None\n    relu_2__paper = torch.nn.functional.relu(bns_1__paper, inplace = True);  bns_1__paper = None\n    relu_2__author = torch.nn.functional.relu(bns_1__author, inplace = True);  bns_1__author = None\n    relu_2__institution = torch.nn.functional.relu(bns_1__institution, inplace = True);  bns_1__institution = None\n    relu_2__field_of_study = torch.nn.functional.relu(bns_1__field_of_study, inplace = True);  bns_1__field_of_study = None\n    dropout_2__paper = torch.nn.functional.dropout(relu_2__paper, p = 0.5, training = True, inplace = False);  relu_2__paper = None\n    dropout_2__author = torch.nn.functional.dropout(relu_2__author, p = 0.5, training = True, inplace = False);  relu_2__author = None\n    dropout_2__institution = torch.nn.functional.dropout(relu_2__institution, p = 0.5, training = True, inplace = False);  relu_2__institution = None\n    dropout_2__field_of_study = torch.nn.functional.dropout(relu_2__field_of_study, p = 0.5, training = True, inplace = False);  relu_2__field_of_study = None\n    fc_out__paper = self.fc_out.paper(dropout_2__paper);  dropout_2__paper = None\n    fc_out__author = self.fc_out.author(dropout_2__author);  dropout_2__author = None\n    fc_out__institution = self.fc_out.institution(dropout_2__institution);  dropout_2__institution = None\n    fc_out__field_of_study = self.fc_out.field_of_study(dropout_2__field_of_study);  dropout_2__field_of_study = None\n    return {\'paper\': fc_out__paper, \'author\': fc_out__author, \'institution\': fc_out__institution, \'field_of_study\': fc_out__field_of_study}\n    '
_forward_hooks: OrderedDict([])
_forward_pre_hooks: OrderedDict([])
_get_backward_hooks: <bound method Module._get_backward_hooks of GraphModule(
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
)>
_get_name: <bound method Module._get_name of GraphModule(
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
)>
_graph: <torch.fx.graph.Graph object at 0x7f690e09bcd0>
_is_full_backward_hook: None
_load_from_state_dict: <bound method Module._load_from_state_dict of GraphModule(
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
)>
_load_state_dict_pre_hooks: OrderedDict([])
_maybe_warn_non_full_backward_hook: <bound method Module._maybe_warn_non_full_backward_hook of GraphModule(
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
)>
_modules: OrderedDict([('convs', ModuleList(
  (0): M...s=4)
  )
)), ('lins', ModuleList(
  (0): M...rue)
  )
)), ('bns', ModuleList(
  (0): M...rue)
  )
)), ('fc_out', ModuleDict(
  (paper...as=True)
))])
_named_members: <bound method Module._named_members of GraphModule(
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
)>
_non_persistent_buffers_set: {}
_parameters: OrderedDict([])
_register_load_state_dict_pre_hook: <bound method Module._register_load_state_dict_pre_hook of GraphModule(
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
)>
_register_state_dict_hook: <bound method Module._register_state_dict_hook of GraphModule(
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
)>
_replicate_for_data_parallel: <bound method Module._replicate_for_data_parallel of GraphModule(
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
)>
_save_to_state_dict: <bound method Module._save_to_state_dict of GraphModule(
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
)>
_slow_forward: <bound method Module._slow_forward of GraphModule(
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
)>
_state_dict_hooks: OrderedDict([])
_version: 1

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