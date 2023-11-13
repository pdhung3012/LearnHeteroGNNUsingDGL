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

dataset_pg = dgl.data.CSVDataset('./csvGraph4IO_train/',force_reload=True)
# dataset_pg = dgl.data.CSVDataset('./csvGraph4IO/train/')
# dataset_ll = torch.load('./graph_list_benchmark_test_all_pg_plus_rodinia.pt')

test_idx = list(range(0,4))
train_idx = list(range(0,4))
ind_count = 0

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
        # print(rel_names)
        # print('out feat {}'.format(out_feats))
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
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
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv4(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv5(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv6(graph, h)
        return h


class HeteroRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, g):
        h = g.ndata['feat']
        # print('h feat {}'.format(h))
        h = self.rgcn(g, h)
        # print('damn h for g {} \n ge herre {}'.format(g, h))
        # print('h shape {}'.format(h))
        # input('bbbb')
        # if len(list(h2.keys()))>0:
        #     h=h2
        with g.local_scope():
            g.ndata['h'] = h
            hg = 0

            for ntype in g.ntypes:
                # print('ntyoe {}'.format(ntype))
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
                # pass
            # print('type hg {} {} {}'.format(type(hg),hg.shape,hg))
            # input('aaaa')
            return self.regressor(hg)

whole_exp = 0

prev_max_num_correct = -1000

flag_15 = 0

num_examples = len(dataset_pg)

output_final = []
label_final = []

print('train_idx {}\ntest_idx {}'.format(train_idx,test_idx))
train_sampler = SubsetRandomSampler(train_idx)
dataset_test = Subset(dataset_pg, test_idx)


etypes = [('control', 'control', 'control'), ('control', 'call', 'control'), ('control', 'data', 'variable'), ('variable', 'data', 'control')]
# , ('variable', 'data', 'control')
# class_names = ['Private Clause', 'Reduction Clause']
fopResult='/home/hungphd/media/aiio/results/lightGBM_subset/'
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
    nameEdge = 'edge-{}-{}'.format(colCurrent, colNext)
    dictEdges[nameEdge]=(colCurrent,nameEdge,colNext)
    # nameEdgeReverse = '{}_BA_{}'.format(colNext,colCurrent)
    # dictEdges[nameEdgeReverse] = (colNext, nameEdgeReverse, colCurrent)

etypes=list(dictEdges.values())


train_dataloader_pg = GraphDataLoader(dataset_pg, shuffle=False, batch_size=100)
test_dataloader_pg = GraphDataLoader(dataset_test, batch_size=100)


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
cross_entropy_loss = nn.CrossEntropyLoss()
best_loss=10000
fpBestModel='./best_model.pt'
for epoch in range(20):
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

