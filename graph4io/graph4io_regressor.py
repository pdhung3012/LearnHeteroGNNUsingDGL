import traceback

import dgl
import torch
import torch.nn.functional as F
import numpy as np
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import dgl.nn.pytorch as dglnn
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from torch.utils.data import Subset
import tensorflow as tf
from sklearn.metrics import *

dataset_pg_train = dgl.data.CSVDataset('./csvGraph4IO_train/',force_reload=True)
dataset_pg_test = dgl.data.CSVDataset('./csvGraph4IO_test/',force_reload=True)
# dataset_ll = torch.load('./graph_list_benchmark_test_all_pg_plus_rodinia.pt')
#
test_idx = list(range(0,100))
train_idx = list(range(0,700))
# ind_count = 0
#
# for data_ll in dataset_ll:
#     ll_file_name = data_ll[3]['file_name'].split('/')[3]
#     class_name = int(data_ll[3]['file_name'].split('/')[2])
#     if 'rod--3.1' in ll_file_name or 'heartwall-main-' in ll_file_name:
#         test_idx.append(ind_count)
#     elif class_name == 0 or class_name == 2:
#         train_idx.append(ind_count)
#     ind_count += 1

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        # print('out_features {}'.format(out_feats))
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        print('rels 1 {}'.format(rel_names))
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        print('rels 2 {}'.format(rel_names))
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        print('rels 3 {}'.format(rel_names))
        self.conv4 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv5 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv6 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs is features of nodes
        # print('graph {}\nimputs {}'.format(graph,inputs))
        # print(' 1 h shape {}'.format(inputs.keys()))
        h = self.conv1(graph, inputs)
        # print(' 1 h shape {}'.format(h.keys()))
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        # print(' 2 h shape {}'.format(h.keys()))
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        # print(' 3 h shape {}'.format(h))
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv4(graph, h)
        # print(' 4 h shape {}'.format(h))
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv5(graph, h)
        # print(' 5 h shape {}'.format(h))
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv6(graph, h)
        # print(' 6 h shape {}'.format(h.keys()))
        # print(' 6 h shape {}'.format(h['col10'].shape))
        # input('bbb ')
        return h


class HeteroRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, g):
        h = g.ndata['feat']
        # print('h feat {}'.format(h.keys()))
        h = self.rgcn(g, h)
        # print('damn h for g {} \n ge herre {}'.format(g, h.keys()))
        # input('bbb')
        # if len(list(h2.keys()))>0:
        #     h=h2
        with g.local_scope():
            # print('go inside')
            # try:
            g.ndata['h'] = h
            # print('g.ndata[h] {}'.format(g.ndata['h']))
            hg = 0
            # print('damn {}'.format(g.ndata['h']))
            for ntype in g.ntypes:
                # print('info ntypes {}'.format(ntype))
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
                # print('end info ntypes {} {}'.format(ntype,hg.shape))
            # print('type hg {} {} {}'.format(type(hg), hg.shape, hg))
            # input('aaaa')
            # except Exception as e:
            #     traceback.print_exc()
                # return torch.Tensor(0.0)

                # pass
        retValue=self.regressor(hg)
        # print('end info ntypes {}'.format(ntype))
        return retValue

whole_exp = 0

prev_max_num_correct = -1000

flag_15 = 0

num_examples = len(dataset_pg_train)

output_final = []
label_final = []

train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)
# dataset_test = Subset(dataset_pg, test_idx)
fopResult='/home/hungphd/media/aiio/results/lightGBM_subset/'

# lstSubsetFeaturesPositive=list(range(0,45))
# lstSubsetFeaturesPositive.remove(2)
# lstSubsetFeaturesPositive.remove(3)
# lstSubsetFeaturesPositive.remove(4)
# lstSubsetFeaturesPositive.remove(5)
# lstSubsetFeaturesPositive.remove(6)
# lstSubsetFeaturesPositive.remove(1)
# lstSubsetFeaturesPositive.remove(11)
# lstSubsetFeaturesPositive.remove(12)
lstSubsetFeaturesPositive=[9,12,13]



fpStat=fopResult+'stat.txt'
f1=open(fpStat,'r')
arrLines=f1.read().split('\n')
f1.close()
lstHeaderCols=[]
lstHeaderIndexes=[]
lstFullHeaderCols=[]
for i in range(1,len(arrLines)-1):
    # colName=arrLines[i].split('\t')[0]
    colName = 'col{}'.format(i)
    realIndex=i-1
    lstFullHeaderCols.append(colName)
    if realIndex in lstSubsetFeaturesPositive:
        lstHeaderCols.append(colName)
        lstHeaderIndexes.append(realIndex)

dictEdges={}
for i in range(0,len(lstHeaderCols)-1):
    # colCurrent=lstHeaderCols[i]
    # colNext=lstHeaderCols[i+1]
    # nameEdge='{}_AB_{}'.format(colCurrent,colNext)
    colCurrent = lstHeaderCols[i]
    colNext = lstHeaderCols[i+1]
    nameEdge = 'edge-{}-{}'.format(colCurrent, colNext).replace('-col','')
    dictEdges[nameEdge]=(colCurrent,nameEdge,colNext)
    nameEdgeReverse = 'edge-{}-{}'.format(colNext,colCurrent).replace('-col','')
    dictEdges[nameEdgeReverse] = (colNext, nameEdgeReverse, colCurrent)

etypes=list(dictEdges.values())
print(etypes)
# etypes = [('control', 'control', 'control'), ('control', 'call', 'control'), ('control', 'data', 'variable'), ('variable', 'data', 'control')]
# class_names = ['Private Clause', 'Reduction Clause']
train_sampler = SubsetRandomSampler(train_idx)
dataset_test = Subset(dataset_pg_test, test_idx)

train_dataloader_pg = GraphDataLoader(dataset_pg_train, shuffle=False, batch_size=100, sampler=train_sampler)
test_dataloader_pg = GraphDataLoader(dataset_test, shuffle=False, batch_size=100)


model_pg = HeteroRegressor(120, 64, 1, etypes)

# model_pg = torch.load('./model-rodinia-best.pt')
# opt = torch.optim.Adam(model_pg.parameters(), lr=0.01)
# total_loss = 0
# loss_list = []
# epoch_list = []
#
# num_correct = 0
# num_tests = 0
# total_pred = []
# total_label = []


opt = torch.optim.Adam(model_pg.parameters())
# cross_entropy_loss = nn.CrossEntropyLoss()
best_loss=10000
fpBestModel='./best_model.pt'
numEpochs=2000
for epoch in range(numEpochs):
    # lstPredicts=[]
    # lstLabels=[]
    tsPredicts = None
    tsLabels = None
    for batched_graph, labels in train_dataloader_pg:
        # feats = batched_graph.ndata['attr']
        # print(feats)

        logits = model_pg(batched_graph)
        # print(logits)
        predicts=logits.reshape([-1, 1]).float()
        # predicts.requires_grad=True
        labels=labels.float()
        labels.requires_grad=True
        predicts=torch.reshape(predicts, [-1])
        labels=torch.reshape(labels, [-1])
        # print('{} {}'.format(predicts.shape,labels.shape))
        # print(predicts)
        # print(labels)
        if tsPredicts is None:
            tsPredicts=predicts
            tsLabels=labels
        else:
            tsPredicts=torch.cat((tsPredicts,predicts),0)
            tsLabels = torch.cat((tsLabels, labels), 0)
    # tsPredicts=torch.tensor(lstPredicts,requires_grad=True,dtype=float)
    # tsLabels = torch.tensor(lstLabels, requires_grad=True,dtype=float)
    # print(tsPredicts.shape)
    # loss = cross_entropy_loss(tsPredicts, tsLabels)
    loss = F.l1_loss(tsPredicts, tsLabels)
    opt.zero_grad()
    loss.backward()
    opt.step()
    isSave=False
    if best_loss>loss:
        best_loss=loss
        isSave=True
        torch.save(model_pg,fpBestModel)

    print('end epoch {} with loss {} (is_save {} with best loss {})'.format(epoch+1,loss,isSave,best_loss))
        # print('go here')

model_pg = torch.load(fpBestModel)
opt = torch.optim.Adam(model_pg.parameters(), lr=0.01)
total_loss = 0
loss_list = []
epoch_list = []

num_correct = 0
num_tests = 0
total_pred = []
total_label = []

for batched_graph, labels in test_dataloader_pg:
    pred = model_pg(batched_graph)

    pred_numpy = pred.detach().numpy()

    # for ind_pred, ind_label in zip(pred_numpy, labels):
    #     if np.argmax(ind_pred) == ind_label:
    #         num_correct += 1
    #     total_pred.append(np.argmax(ind_pred))
    total_pred.extend(pred_numpy)

    # num_tests += len(labels)

    label_tmp = labels.data.cpu().numpy()
    total_label.extend(label_tmp)
    #
    # label_final = labels
    # output_final = total_pred

# print('num correct: ', num_correct)
# print(classification_report(total_label, total_pred, target_names=class_names))
# cf_matrix = confusion_matrix(total_label, total_pred)
# print(cf_matrix)
print('MAE {}'.format(mean_absolute_error(total_label,total_pred)))

