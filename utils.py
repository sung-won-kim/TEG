import numpy as np
import scipy.sparse as sp
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score
from torch_geometric.datasets import Coauthor
import torch_geometric
import scipy.sparse as sp
import torch
import scipy.io as sio
import pickle

valid_num_dic = {"Amazon_clothing": 17, "Amazon_electronics": 36, "dblp": 27}


def load_data(dataset_source):
    if dataset_source in ['Amazon_clothing', 'Amazon_electronics', 'dblp']:
        n1s = []
        n2s = []
        for line in open(f"./dataset/{dataset_source}/{dataset_source}_network"):
            n1, n2 = line.strip().split("\t")
            n1s.append(int(n1))
            n2s.append(int(n2))

        edges = torch.LongTensor([n1s, n2s])

        num_nodes = max(max(n1s), max(n2s)) + 1
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                            shape=(num_nodes, num_nodes))

        data_train = sio.loadmat(
            f"./dataset/{dataset_source}/{dataset_source}_train.mat")
        train_class = list(
            set(data_train["Label"].reshape((1, len(data_train["Label"])))[0])
        )

        data_test = sio.loadmat(
            f"./dataset/{dataset_source}/{dataset_source}_test.mat")
        class_list_test = list(
            set(data_test["Label"].reshape((1, len(data_test["Label"])))[0])
        )

        labels = np.zeros((num_nodes, 1))
        labels[data_train["Index"]] = data_train["Label"]
        labels[data_test["Index"]] = data_test["Label"]

        features = np.zeros((num_nodes, data_train["Attributes"].shape[1]))
        features[data_train["Index"]] = data_train["Attributes"].toarray()
        features[data_test["Index"]] = data_test["Attributes"].toarray()

        class_list = []
        for cla in labels:
            if cla[0] not in class_list:
                class_list.append(cla[0])

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels):
            id_by_class[cla[0]].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)

        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)

        class_list_valid = random.sample(
            train_class, valid_num_dic[dataset_source])

        class_list_train = list(
            set(train_class).difference(set(class_list_valid)))

    elif dataset_source == 'corafull':
        cora_full = torch_geometric.datasets.CitationFull(
            './dataset', 'cora')

        edges = cora_full.data.edge_index

        num_nodes = max(max(edges[0]), max(edges[1])) + 1

        adj = sp.coo_matrix((np.ones(len(edges[0])), (edges[0], edges[1])),
                            shape=(num_nodes, num_nodes))

        features = cora_full.data.x
        labels = cora_full.data.y

        class_list = cora_full.data.y.unique().tolist()

        with open(file='./dataset/cora/cls_split.pkl', mode='rb') as f:
            class_list_train, class_list_valid, class_list_test = pickle.load(f)

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.tolist()):
            id_by_class[cla].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)

        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)

    elif dataset_source == 'coauthorCS':
        CS = Coauthor(root='./dataset/CS', name='CS')
        data = CS.data

        edges = data.edge_index

        num_nodes = max(max(edges[0]), max(edges[1])) + 1

        adj = sp.coo_matrix((np.ones(len(edges[0])), (edges[0], edges[1])),
                            shape=(num_nodes, num_nodes))

        features = data.x
        labels = data.y

        class_list = data.y.unique().tolist()

        with open(file='./dataset/CS/cls_split.pkl', mode='rb') as f:
            class_list_train, class_list_valid, class_list_test = pickle.load(f)

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.tolist()):
            id_by_class[cla].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)

        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)


    elif dataset_source == 'ogbn-arxiv':

        from ogb.nodeproppred import NodePropPredDataset

        dataset = NodePropPredDataset(name = 'ogbn-arxiv')

        split_idx = dataset.get_idx_split()
        # train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, labels = dataset[0] # graph: library-agnostic graph object

        n1s=graph['edge_index'][0]
        n2s=graph['edge_index'][1]

        edges = torch.LongTensor(graph['edge_index'])

        num_nodes = graph['num_nodes']

        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                        shape=(num_nodes, num_nodes))    
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        features=torch.FloatTensor(graph['node_feat'])
        labels=torch.LongTensor(labels).squeeze()

        class_list = labels.unique().tolist()

        with open(file='./dataset/ogbn_arxiv/cls_split.pkl', mode='rb') as f:
            class_list_train, class_list_valid, class_list_test = pickle.load(f)

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.tolist()):
            id_by_class[cla].append(id)


    elif dataset_source == 'citeseer':
        citeseer = torch_geometric.datasets.Planetoid(
            './dataset', 'CiteSeer')

        edges = citeseer.data.edge_index

        num_nodes = max(max(edges[0]), max(edges[1])) + 1

        adj = sp.coo_matrix((np.ones(len(edges[0])), (edges[0], edges[1])),
                            shape=(num_nodes, num_nodes))

        features = citeseer.data.x
        labels = citeseer.data.y

        class_list = citeseer.data.y.unique().tolist()

        with open(file='./dataset/CiteSeer/cls_split.pkl', mode='rb') as f:
            class_list_train, class_list_valid, class_list_test = pickle.load(f)

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.tolist()):
            id_by_class[cla].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)

        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)


    return (
        edges,
        adj,
        features,
        labels,
        degree,
        class_list_train,
        class_list_valid,
        class_list_test,
        id_by_class,
        num_nodes
    )


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / output.shape[0]


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average="weighted")
    return f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def seed_everything(seed=0):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def task_generator_in_class(
    id_by_class, selected_class_list, n_way, k_shot, m_query
):
    # sample class indices
    class_selected = selected_class_list
    id_support = []
    id_query = []

    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])

    # return [0] (k-shot x n_way) support data id array
    #        [1] (n_query x n_way) query data id array
    #        [2] (n_way) selected class list
    return np.array(id_support), np.array(id_query), class_selected


def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M
