#! /usr/bin/env python3

from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter

import sys

# from args import *
from model import *
from utils import *
from dataset import *

if not os.path.isdir('results'):
    os.mkdir('results')
# args

def args():
    pass

args.comment = '0'
args.task = 'link'
args.model = 'PGNN'
args.gpu = 'store_true'
#args.dataset = "communities"

#../staging/edge_gt20_C8_microC_1Mto1100K.txt 
#../thresholding/gt20_labels.txt

args.dataset = sys.argv[1]#"/home/spmoran/temp_spmoran/torch2/microC_pilot_redux/single_redux/staging/edge_gt20_C8_microC_1Mto1100K.txt"
args.labelpath = sys.argv[2]#'/home/spmoran/temp_spmoran/torch2/microC_pilot_redux/single_redux/thresholding/gt20_labels.txt'
args.cache = 'store_false'
args.cuda = '0'

args.normalize_adj = 'store_true'
args.feature_pre = 'store_false'
args.remove_link_ratio = 0.2

args.batch_size = 8 #8
args.layer_num = 3
args.feature_dim = 64 #32
args.hidden_dim = 64 #32
args.output_dim = 64 #32
args.anchor_num = 128 #64

#args.batch_size = 8
#args.layer_num = 3
#args.feature_dim = 32
#args.hidden_dim = 32
#args.output_dim = 32
#args.anchor_num = 64

args.lr = 1e-2
args.epoch_num = 2001
args.repeat_num = 2
args.epoch_log = 10

args.gpu = True
args.task = 'link'
args.model = 'PGNN'
args.cache = False
args.rm_feature = False
args.permute = True
args.feature_pre = True
args.dropout = True
args.approximate = -1
args.normalize_adj = False

print(args.__dict__)
np.random.seed(123)
np.random.seed()
writer_train = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_train')
writer_val = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_val')
writer_test = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_test')


# set up gpu
if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
else:
    print('Using CPU')


device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')


args.cache = False
dataset_name = args.dataset
name_frag=dataset_name.split('/')[-1].split('.txt')[0]

results = []
# out_embeddings = dict()
# data_embeddings = dict()
# init_data = dict()

# for repeat in range(args.repeat_num):
result_val = []
result_test = []

time1 = time.time()
data_in = get_tg_dataset(args, dataset_name, use_cache=args.cache, remove_feature=args.rm_feature)
time2 = time.time()
print(dataset_name, 'load time',  time2-time1)

num_features = data_in.x.shape[1]
num_node_classes = None
num_graph_classes = None
if 'y' in data_in.__dict__ and data_in.y is not None:
    num_node_classes = data_in.y.max().item()+1
    #

if 'y_graph' in data_in.__dict__ and data_in.y_graph is not None:
    num_graph_classes = data_in.y_graph.numpy()[0]+1
    #

print('Dataset', dataset_name, 'Graph', len(data_in), 'Feature', num_features, 'Node Class', num_node_classes, 'Graph Class', num_graph_classes)
nodes = data_in.num_nodes
edges = data_in.num_edges
print('Node: {}'.format(nodes))
print('Edge: {}'.format(edges))

args.batch_size = args.batch_size
print('Anchor num {}, Batch size {}'.format(args.anchor_num, args.batch_size))

# data
preselect_anchor(data_in, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cpu')
data_in = data_in.to(device)

init_data = data_in
input_dim = num_features
output_dim = args.output_dim
model = locals()[args.model](input_dim=input_dim, feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim, output_dim=output_dim,
            feature_pre=args.feature_pre, layer_num=args.layer_num, dropout=args.dropout).to(device)    
# loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
if 'link' in args.task:
    loss_func = nn.BCEWithLogitsLoss()
    out_act = nn.Sigmoid()

for epoch in range(args.epoch_num):
    if epoch==200:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
    model.train()
    optimizer.zero_grad()
    # for id, data in enumerate(data_list[:effective_len]):
#
    if args.permute:
        preselect_anchor(data_in, layer_num=args.layer_num, anchor_num=args.anchor_num, device=device)
#        
    out = model(data_in)
    # get_link_mask(data,resplit=False)  # resample negative links
    edge_mask_train = np.concatenate((data_in.mask_link_positive_train, data_in.mask_link_negative_train), axis=-1)
    nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0,:]).long().to(device))
    nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1,:]).long().to(device))
    pred = torch.sum(nodes_first * nodes_second, dim=-1)
    label_positive = torch.ones([data_in.mask_link_positive_train.shape[1],], dtype=pred.dtype)
    label_negative = torch.zeros([data_in.mask_link_negative_train.shape[1],], dtype=pred.dtype)
    label = torch.cat((label_positive,label_negative)).to(device)
    loss = loss_func(pred, label)
    # update
    loss.backward()
    if args.batch_size>1:
        # if this is slow, no need to do this normalization
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= args.batch_size
    optimizer.step()
    optimizer.zero_grad()
    ##
    if epoch % args.epoch_log == 0:
        # evaluate
        model.eval()
        loss_train = 0
        loss_val = 0
        loss_test = 0
        correct_train = 0
        all_train = 0
        correct_val = 0
        all_val = 0
        correct_test = 0
        all_test = 0
        auc_train = 0
        auc_val = 0
        auc_test = 0
        emb_norm_min = 0
        emb_norm_max = 0
        emb_norm_mean = 0
        # for id, data in enumerate(data_list):
        out = model(data_in)
        # TEST PROC exporting variables
        if epoch == args.epoch_num-1:
            print(f"epoch: {epoch}")

            # fdump_name = f"{name_frag}_initgraph.pkl"
            # with open(fdump_name, 'wb') as fd:
            #     pickle.dump(init_data, fd)
            # print(f"write graph to {fdump_name}")

            fdump_name = f"{name_frag}_outANDdata.pkl"
            with open(fdump_name, 'wb') as fd:
                pickle.dump([out, data_in], fd)
            print(f"write out and data epochs to {fdump_name}")
#
        emb_norm_min += torch.norm(out.data, dim=1).min().cpu().numpy()
        emb_norm_max += torch.norm(out.data, dim=1).max().cpu().numpy()
        emb_norm_mean += torch.norm(out.data, dim=1).mean().cpu().numpy()
#
        # train
        # get_link_mask(data, resplit=False)  # resample negative links
        edge_mask_train = np.concatenate((data_in.mask_link_positive_train, data_in.mask_link_negative_train), axis=-1)
        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0, :]).long().to(device))
        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1, :]).long().to(device))
        pred = torch.sum(nodes_first * nodes_second, dim=-1)
        label_positive = torch.ones([data_in.mask_link_positive_train.shape[1], ], dtype=pred.dtype)
        label_negative = torch.zeros([data_in.mask_link_negative_train.shape[1], ], dtype=pred.dtype)
        label = torch.cat((label_positive, label_negative)).to(device)
        loss_train += loss_func(pred, label).cpu().data.numpy()
        auc_train += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
        # val
        edge_mask_val = np.concatenate((data_in.mask_link_positive_val, data_in.mask_link_negative_val), axis=-1)
        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[0, :]).long().to(device))
        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[1, :]).long().to(device))
        pred = torch.sum(nodes_first * nodes_second, dim=-1)
        label_positive = torch.ones([data_in.mask_link_positive_val.shape[1], ], dtype=pred.dtype)
        label_negative = torch.zeros([data_in.mask_link_negative_val.shape[1], ], dtype=pred.dtype)
        label = torch.cat((label_positive, label_negative)).to(device)
        loss_val += loss_func(pred, label).cpu().data.numpy()
        auc_val += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
        # test
        edge_mask_test = np.concatenate((data_in.mask_link_positive_test, data_in.mask_link_negative_test), axis=-1)
        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[0, :]).long().to(device))
        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[1, :]).long().to(device))
        pred = torch.sum(nodes_first * nodes_second, dim=-1)
        label_positive = torch.ones([data_in.mask_link_positive_test.shape[1], ], dtype=pred.dtype)
        label_negative = torch.zeros([data_in.mask_link_negative_test.shape[1], ], dtype=pred.dtype)
        label = torch.cat((label_positive, label_negative)).to(device)
        loss_test += loss_func(pred, label).cpu().data.numpy()
        auc_test += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
#
#
        print(epoch, 'Loss {:.4f}'.format(loss_train), 'Train AUC: {:.4f}'.format(auc_train),
              'Val AUC: {:.4f}'.format(auc_val), 'Test AUC: {:.4f}'.format(auc_test))
        writer_train.add_scalar( 'auc_'+dataset_name, auc_train, epoch)
        writer_train.add_scalar( 'loss_'+dataset_name, loss_train, epoch)
        writer_val.add_scalar('auc_'+dataset_name, auc_val, epoch)
        writer_train.add_scalar( '/loss_'+dataset_name, loss_val, epoch)
        writer_test.add_scalar('auc_'+dataset_name, auc_test, epoch)
        writer_test.add_scalar('loss_'+dataset_name, loss_test, epoch)
        writer_test.add_scalar('emb_min_'+dataset_name, emb_norm_min, epoch)
        writer_test.add_scalar('emb_max_'+dataset_name, emb_norm_max, epoch)
        writer_test.add_scalar('emb_mean_'+dataset_name, emb_norm_mean, epoch)
        result_val.append(auc_val)
        result_test.append(auc_test)


result_val = np.array(result_val)
result_test = np.array(result_test)
results.append(result_test[np.argmax(result_val)])


results = np.array(results)
results_mean = np.mean(results).round(6)
results_std = np.std(results).round(6)
print('-----------------Final-------------------')
print(results_mean, results_std)



with open('results/{}_{}_{}_layer{}_approximate{}.txt'.format(args.task,args.model,name_frag,args.layer_num,args.approximate), 'w') as f:
    f.write('{}, {}\n'.format(results_mean, results_std))

# export scalar data to JSON for external processing
writer_train.export_scalars_to_json("./all_scalars.json")
writer_train.close()
writer_val.export_scalars_to_json("./all_scalars.json")
writer_val.close()
writer_test.export_scalars_to_json("./all_scalars.json")
writer_test.close()


