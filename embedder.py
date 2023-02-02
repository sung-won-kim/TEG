import torch
import torch.nn as nn
import os
from argument import config2string
import numpy as np
from utils import *
import networkx as nx


class embedder(nn.Module):
    def __init__(self, args, conf, set_seed):
        super().__init__()

        self.args = args
        self.conf = conf
        self.set_seed = set_seed
        self.config_str = config2string(args)
        print("\n[Config] {}\n".format(self.config_str))

        # Select GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self.device = f'cuda:{args.device}' if torch.cuda.is_available(
        ) else 'cpu'
        torch.cuda.set_device(self.device)

        self.edges, self.adj, self.features, self.labels, self.degrees, self.class_list_train, self.class_list_valid, self.class_list_test, self.id_by_class, self.num_nodes = load_data(
            args.dataset)
        self.edges = self.edges.to(self.device)
        self.adj = self.adj.to(self.device)
        self.features = self.features.to(self.device)
        self.labels = self.labels.to(self.device)
        self.degrees = self.degrees.to(self.device)

        # _________________________
        # calculates shortest dists
        self.edges_hub = self.edges
        print("Generating a structural feature...")
        for i in range(self.args.anchor_size):
            hub_index = len(self.features) + i
            num_sample_node = int(len(self.features)/(2**(i+1)))
            if num_sample_node < 1:
                print(f'   Virtual Anchor Node [{i+1}] samples less than 1')
                num_sample_node = 1
            selected_nodes_with_hub_i = random.sample(
                list(range(len(self.features))), num_sample_node)
            edge_hub_i = torch.LongTensor(
                [hub_index] * len(selected_nodes_with_hub_i))
            edge_hub_j = torch.LongTensor(
                selected_nodes_with_hub_i)
            edge1_bi = torch.cat([edge_hub_i, edge_hub_j])
            edge2_bi = torch.cat([edge_hub_j, edge_hub_i])
            edge_index_hub_i = torch.stack(
                (edge1_bi, edge2_bi)).to(self.device)
            self.edges_hub = torch.cat(
                [self.edges_hub, edge_index_hub_i], 1)
        print("Done.\n")

        graph = nx.Graph()
        edge_list = self.edges_hub.transpose(1, 0).tolist()
        graph.add_edges_from(edge_list)

        structural_feature = []
        for i in range(self.args.anchor_size):
            hub_index = len(self.features) + i
            hub_i_feature = []
            spd_i = nx.single_source_shortest_path_length(
                graph, hub_index)
            for j in range(len(self.features)):
                try:
                    hub_spd_ij = spd_i[j]
                except:
                    hub_spd_ij = np.inf
                hub_spd_ij = 1 / (hub_spd_ij+1)
                hub_i_feature.append(hub_spd_ij)
            structural_feature.append(hub_i_feature)

        self.structural_features = torch.Tensor(
            structural_feature).T.to(self.device)

        self.n_way = args.way
        self.k_shot = args.shot
        self.n_query = args.qry
