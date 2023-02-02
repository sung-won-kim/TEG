from calendar import c
from layers.GCN import GCN
from layers.EGNN import EGNN
from embedder import embedder
from tqdm.auto import tqdm
from utils import *
from torch import optim
import torch.nn.functional as F
import torch
from argument import config2string, parse_args


class teg_trainer(embedder):
    def __init__(self, args, conf, set_seed):
        embedder.__init__(self, args, conf, set_seed)
        self.conv = GCN(self.features.shape[1],
                        conf['gcn_out'], args.dropout).to(self.device)
        self.egnn = EGNN(self.structural_features.shape[1], conf['egnn_in'],
                         n_layers=args.n_layers).to(self.device)

        self.optim = optim.Adam([
            {'params': self.conv.parameters()},
            {'params': self.egnn.parameters()}],
            lr=args.lr, weight_decay=5e-4
        )

        self.config_str = config2string(args)
        self.set_seed = set_seed

    def train_epoch(self, mode, n_episode, epoch):

        loss_fn = torch.nn.NLLLoss()

        if mode == 'train':
            if epoch != 0:
                self.conv.train()
                self.egnn.train()
            else:
                self.conv.eval()
                self.egnn.eval()

        else:
            self.conv.eval()
            self.egnn.eval()

        if mode == 'train' or mode == 'valid':
            loss_epoch = 0

        acc_epoch = []
        f1_epoch = []

        for episode in range(n_episode):

            if mode == 'train':
                self.optim.zero_grad()

            if mode == 'train':
                class_selected = random.sample(
                    self.class_list_train, self.args.way)

            elif mode == 'valid':
                class_selected = random.sample(
                    self.class_list_valid, self.args.way)

            elif mode == 'test':
                class_selected = random.sample(
                    self.class_list_test, self.args.way)

            id_support, id_query, class_selected = task_generator_in_class(
                self.id_by_class, class_selected, self.n_way, self.k_shot, self.n_query)

            # ________________
            # graph conv (GCN)
            embeddings = self.conv(self.features, self.edges)

            # _____________
            # Task sampling
            embeds_spt = embeddings[id_support]
            embeds_qry = embeddings[id_query]

            embeds_epi = torch.cat([embeds_spt, embeds_qry])

            # structural features
            spt_str = self.structural_features[id_support]
            qry_str = self.structural_features[id_query]

            epi_str = torch.cat([spt_str, qry_str])

            # _______________________________
            # calculate a graph embedder loss
            gcn_spt = embeds_epi[:len(id_support), :]
            gcn_qry = embeds_epi[len(id_support):, :]
            gcn_spt = gcn_spt.view(
                [self.n_way, self.k_shot, gcn_spt.shape[1]])
            proto_embeds_gcn = gcn_spt.mean(1)
            dists_gcn = euclidean_dist(
                gcn_qry, proto_embeds_gcn)
            output_gcn = F.log_softmax(-dists_gcn, dim=1)

            # ___________________
            # Task-specific graph
            edge1 = []
            for i in range(len(id_support), len(id_support)+len(id_query)):
                temp = [i] * len(id_support)
                edge1.extend(temp)
            edge1 = torch.LongTensor(edge1)
            edge2 = torch.LongTensor(
                list(range(len(id_support))) * len(id_query))
            edge1_bi = torch.cat([edge1, edge2])
            edge2_bi = torch.cat([edge2, edge1])
            edge_index = torch.stack((edge1_bi, edge2_bi)).to(self.device)

            # ______________________
            # EGNN - Task adaptation
            epi_str, embeds_epi = self.egnn(epi_str, embeds_epi, edge_index)

            # __________
            # Prototypes
            embeds_spt = embeds_epi[:len(id_support), :]
            embeds_qry = embeds_epi[len(id_support):, :]

            embeds_spt = embeds_spt.view(
                [self.n_way, self.k_shot, embeds_spt.shape[1]])

            embeds_proto = embeds_spt.mean(1)

            # __________
            # Prediction
            dists_output = euclidean_dist(
                embeds_qry, embeds_proto)
            output = F.log_softmax(-dists_output, dim=1)
            output_softmax = F.softmax(-dists_output, dim=1)

            # _________________________
            # Relabeling for meta-tasks
            label_list = torch.LongTensor(
                [class_selected.index(i)
                    for i in self.labels[id_query]]
            ).to(self.device)

            if mode == 'train' or mode == 'valid':

                # ___________
                # Loss update
                loss_l1_train = loss_fn(
                    output, label_list)  # Network Loss
                loss_l2_train = loss_fn(
                    output_gcn, label_list)  # Graph Embedder Loss

                loss_train = self.args.gamma*loss_l1_train + \
                    (1-self.args.gamma)*loss_l2_train

            if mode == 'train':
                if epoch != 0:
                    loss_train.backward()
                    self.optim.step()
                else:
                    self.optim.zero_grad()

            # ________
            # Accuracy
            output = output_softmax.cpu().detach()
            label_list = label_list.cpu().detach()

            acc_score = accuracy(output, label_list)
            f1_score = f1(output, label_list)

            acc_epoch.append(acc_score)
            f1_epoch.append(f1_score)

        acc_total_epoch = sum(acc_epoch) / len(acc_epoch)
        f1_total_epoch = sum(f1_epoch) / len(f1_epoch)

        if mode == 'train':
            tqdm.write(f"acc_train : {acc_total_epoch:.4f}")

        elif mode == 'valid':
            tqdm.write(f"acc_valid : {acc_total_epoch:.4f}")

        elif mode == 'test':
            tqdm.write(f"acc_test : {acc_total_epoch:.4f}")

        return acc_total_epoch, f1_total_epoch

    def train(self):

        # _____________
        # Best Accuracy
        best_acc_train = 0
        best_f1_train = 0
        best_epoch_train = 0
        best_acc_valid = 0
        best_f1_valid = 0
        best_epoch_valid = 0
        best_acc_test = 0
        best_f1_test = 0
        best_epoch_test = 0

        for epoch in tqdm(range(self.args.epochs+1)):

            acc_train, f1_train = self.train_epoch(
                'train', self.args.episodes, epoch)

            with torch.no_grad():

                acc_valid, f1_valid = self.train_epoch(
                    'valid', self.args.meta_val_num, epoch)

                acc_test, f1_test = self.train_epoch(
                    'test', self.args.meta_test_num, epoch)

            if best_acc_train < acc_train:
                best_acc_train = acc_train
                best_f1_train = f1_train
                best_epoch_train = epoch

            if best_acc_valid < acc_valid:
                best_acc_valid = acc_valid
                best_f1_valid = f1_valid
                best_epoch_valid = epoch

            if best_acc_test < acc_test:
                best_acc_test = acc_test
                best_f1_test = f1_test
                best_epoch_test = epoch

            if acc_valid == best_acc_valid:
                test_acc_at_best_valid = acc_test
                test_f1_at_best_valid = f1_test

            tqdm.write(f"# Current Settings : {self.config_str}")
            tqdm.write(
                f"# Best_Acc_Train : {best_acc_train:.4f}, F1 : {best_f1_train:.4f} at {best_epoch_train} epoch"
            )
            tqdm.write(
                f"# Best_Acc_Valid : {best_acc_valid:.4f}, F1 : {best_f1_valid:.4f} at {best_epoch_valid} epoch"
            )
            tqdm.write(
                f"# Best_Acc_Test : {best_acc_test:.4f}, F1 : {best_f1_test:.4f} at {best_epoch_test} epoch"
            )
            tqdm.write(
                f"# Test_At_Best_Valid : {test_acc_at_best_valid:.4f}, F1 : {test_f1_at_best_valid:.4f} at {best_epoch_valid} epoch\n"
            )

        np.set_printoptions(
            formatter={'float_kind': lambda x: "{0:0.4f}".format(x)})

        return best_acc_train, best_f1_train, best_epoch_train, best_acc_valid, best_f1_valid, best_epoch_valid, best_acc_test, best_f1_test, best_epoch_test, test_acc_at_best_valid, test_f1_at_best_valid
