import os
import gc
import time
import uuid
import argparse
import datetime
import numpy as np

import torch
import torch.nn.functional as F
from util.torch_data_utils import NestedDataLoader
from util.nested_data_utils import nested_gather, nested_map
from model import *
from utils import *
from arch import archs

def main(args):
    if args.seed > 0:
        set_random_seed(args.seed)
    g, init_labels, num_nodes, n_classes, train_nid, val_nid, test_nid, evaluator = load_dataset(args)
    device = "cuda:{}".format(args.gpu) if not args.cpu else 'cpu'

    if args.label_feats and args.num_label_hops >= 3:
        p = f"ogbn-mag_hop{args.num_label_hops}_total_float.pt"
        if os.path.exists(p):
            self_value_dict = torch.load(p)
        else:
            print(args.num_label_hops)
            self_value_dict = propagate_self_value_gpu_parallel(g,'P',max_hop=args.num_label_hops, device=device)
            torch.save(self_value_dict, p)
        print(self_value_dict)
    else:
        self_value_dict = None
    
    import code
    code.interact(local=locals())
    
        
    # =======
    # rearange node idx (for feats & labels)
    # =======
    train_node_nums = len(train_nid)
    valid_node_nums = len(val_nid)
    test_node_nums = len(test_nid)
    trainval_point = train_node_nums
    valtest_point = trainval_point + valid_node_nums
    total_num_nodes = len(train_nid) + len(val_nid) + len(test_nid)

    if total_num_nodes < num_nodes:
        flag = torch.ones(num_nodes, dtype=bool)
        flag[train_nid] = 0
        flag[val_nid] = 0
        flag[test_nid] = 0
        extra_nid = torch.where(flag)[0]
        print(f'Find {len(extra_nid)} extra nid for dataset {args.dataset}')
    else:
        extra_nid = torch.tensor([], dtype=torch.long)

    init2sort = torch.cat([train_nid, val_nid, test_nid, extra_nid])
    sort2init = torch.argsort(init2sort)
    assert torch.all(init_labels[init2sort][sort2init] == init_labels)
    labels = init_labels[init2sort]

    # =======
    # features propagate alongside the metapath
    # =======
    prop_tic = datetime.datetime.now()
    if args.dataset in ['ogbn-proteins', 'ogbn-products']: # homogeneous
        tgt_type = 'hop_0'

        g.ndata['hop_0'] = g.ndata.pop('feat')
        for hop in range(args.num_hops):
            g.update_all(fn.copy_u(f'hop_{hop}', 'm'), fn.mean('m', f'hop_{hop+1}'))
        keys = list(g.ndata.keys())
        print(f'Involved feat keys {keys}')
        feats = {k: g.ndata.pop(k) for k in keys}

    elif args.dataset in ['ogbn-arxiv', 'ogbn-papers100M']: # single-node-type & double-edge-types
        tgt_type = 'P'

        for hop in range(1, args.num_hops + 1):
            for k in list(g.ndata.keys()):
                if len(k) == hop:
                    g['cite'].update_all(
                        fn.copy_u(k, 'msg'),
                        fn.mean('msg', f'm{k}'), etype='cite')
                    g['cited_by'].update_all(
                        fn.copy_u(k, 'msg'),
                        fn.mean('msg', f't{k}'), etype='cited_by')

        keys = list(g.ndata.keys())
        print(f'Involved feat keys {keys}')
        feats = {k: g.ndata.pop(k) for k in keys}

    elif args.dataset == 'ogbn-mag': # multi-node-types & multi-edge-types
        tgt_type = 'P'

        extra_metapath = [] # ['AIAP', 'PAIAP']
        extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_hops + 1]

        print(f'Current num hops = {args.num_hops}')
        if len(extra_metapath):
            max_hops = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
        else:
            max_hops = args.num_hops + 1

        # compute k-hop feature
        g = hg_propagate(g, tgt_type, args.num_hops, max_hops, extra_metapath, echo=False)

        feats = {}
        keys = list(g.nodes[tgt_type].data.keys())
        print(f'Involved feat keys {keys}')
        for k in keys:
            feats[k] = g.nodes[tgt_type].data.pop(k)

        g = clear_hg(g, echo=False)
    else:
        assert 0

    feats = {k: v[init2sort] for k, v in feats.items()}
    
    import code
    code.interact(local=locals())

    feats = {k: v for k, v in feats.items() if k in archs[args.arch][0] or k == tgt_type}

    prop_toc = datetime.datetime.now()
    print(f'Time used for feat prop {prop_toc - prop_tic}')
    gc.collect()
    
    #######################################
    train_h_list_list, train_y = nested_gather([target_h_list_list, init_labels], train_nid)
    valid_h_list_list, valid_y = nested_gather([target_h_list_list, init_labels], val_nid)
    test_h_list_list, test_y = nested_gather([target_h_list_list, init_labels], test_nid)


    train_data_loader = NestedDataLoader(
        [train_h_list_list, train_y],
        batch_size=batch_size, shuffle=True, device=data_loader_device
    )

    valid_data_loader =NestedDataLoader(
        [valid_h_list_list, valid_y], 
        batch_size=batch_size, shuffle=False, device=data_loader_device
    )
    test_data_loader = NestedDataLoader(
        [test_h_list_list, test_y], 
        batch_size=batch_size, shuffle=False, device=data_loader_device
    )


    all_loader = torch.utils.data.DataLoader(
        torch.arange(num_nodes), batch_size=args.batch_size, shuffle=False, drop_last=False)

    checkpt_folder = f'./output/{args.dataset}/'
    if not os.path.exists(checkpt_folder):
        os.makedirs(checkpt_folder)

    if args.amp:
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    device = "cuda:{}".format(args.gpu) if not args.cpu else 'cpu'
    labels_cuda = labels.long().to(device)

    checkpt_file = checkpt_folder + uuid.uuid4().hex
    print(checkpt_file)

    for stage in range(args.start_stage, len(args.stages)):
        epochs = args.stages[stage]

        if len(args.reload):
            pt_path = f'output/ogbn-mag/{args.reload}_{stage-1}.pt'
            assert os.path.exists(pt_path)
            print(f'Reload raw_preds from {pt_path}', flush=True)
            raw_preds = torch.load(pt_path, map_location='cpu')

        # =======
        # Expand training set & train loader
        # =======
        if stage > 0:
            preds = raw_preds.argmax(dim=-1)
            predict_prob = raw_preds.softmax(dim=1)

            train_acc = evaluator(preds[:trainval_point], labels[:trainval_point])
            val_acc = evaluator(preds[trainval_point:valtest_point], labels[trainval_point:valtest_point])
            test_acc = evaluator(preds[valtest_point:total_num_nodes], labels[valtest_point:total_num_nodes])

            print(f'Stage {stage-1} history model:\n\t' \
                + f'Train acc {train_acc*100:.4f} Val acc {val_acc*100:.4f} Test acc {test_acc*100:.4f}')

            confident_mask = predict_prob.max(1)[0] > args.threshold
            val_enhance_offset  = torch.where(confident_mask[trainval_point:valtest_point])[0]
            test_enhance_offset = torch.where(confident_mask[valtest_point:total_num_nodes])[0]
            val_enhance_nid     = val_enhance_offset + trainval_point
            test_enhance_nid    = test_enhance_offset + valtest_point
            enhance_nid = torch.cat((val_enhance_nid, test_enhance_nid))

            print(f'Stage: {stage}, threshold {args.threshold}, confident nodes: {len(enhance_nid)} / {total_num_nodes - trainval_point}')
            val_confident_level = (predict_prob[val_enhance_nid].argmax(1) == labels[val_enhance_nid]).sum() / len(val_enhance_nid)
            print(f'\t\t val confident nodes: {len(val_enhance_nid)} / {valid_node_nums},  val confident level: {val_confident_level}')
            test_confident_level = (predict_prob[test_enhance_nid].argmax(1) == labels[test_enhance_nid]).sum() / len(test_enhance_nid)
            print(f'\t\ttest confident nodes: {len(test_enhance_nid)} / {test_node_nums}, test confident_level: {test_confident_level}')

            del train_loader
            train_batch_size = int(args.batch_size * len(train_nid) / (len(enhance_nid) + len(train_nid)))
            train_loader = torch.utils.data.DataLoader(
                torch.arange(train_node_nums), batch_size=train_batch_size, shuffle=True, drop_last=False)
            enhance_batch_size = int(args.batch_size * len(enhance_nid) / (len(enhance_nid) + len(train_nid)))
            enhance_loader = torch.utils.data.DataLoader(
                enhance_nid, batch_size=enhance_batch_size, shuffle=True, drop_last=False)
        else:
            train_loader = torch.utils.data.DataLoader(
                torch.arange(train_node_nums), batch_size=args.batch_size, shuffle=True, drop_last=False)

        # =======
        # labels propagate alongside the metapath
        # =======
        label_feats = {}
        if args.label_feats:
            if stage > 0:
                label_onehot = predict_prob[sort2init].clone()
            else:
                label_onehot = torch.zeros((num_nodes, n_classes))
            label_onehot[train_nid] = F.one_hot(init_labels[train_nid], n_classes).float()

            if args.dataset in ['ogbn-proteins', 'ogbn-products']: # homogeneous
                g.ndata['s'] = label_onehot
                for hop in range(args.num_label_hops):
                    g.update_all(fn.copy_u('m'*hop+'s', 'msg'), fn.mean('msg', 'm'*(hop+1)+'s'))
                keys = [k[:-1] for k in g.ndata.keys() if k != 's']
                print(f'Involved label keys {keys}')
                label_feats = {k: g.ndata.pop(k+'s') for k in keys}
                g = clear_hg(g, echo=False)

                # remove self effect on label feats
                mm_diag, mmm_diag = torch.load(f'{args.dataset}_diag.pt')
                diag_values = {'mm': mm_diag, 'mmm': mmm_diag}
                for k in diag_values:
                    if k in label_feats:
                        label_feats[k] = label_feats[k] - diag_values[k].unsqueeze(-1) * label_onehot
                        assert torch.all(label_feats[k] > -1e-6)
                        print(k, torch.sum(label_feats[k] < 0), label_feats[k].min())

                condition = lambda ra,rb,rc,k: True
                check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid)

                label_emb = label_feats['mmmmmmmmm']
                # label_emb = (label_feats['m'] + label_feats['mm'] + label_feats['mmm']) / 3
                check_acc({'label_emb': label_emb}, condition, init_labels, train_nid, val_nid, test_nid)

            elif args.dataset in ['ogbn-arxiv', 'ogbn-papers100M']: # single-node-type & double-edge-types
                g.ndata[tgt_type] = label_onehot

                for hop in range(1, args.num_hops + 1):
                    for k in list(g.ndata.keys()):
                        if len(k) == hop:
                            g['cite'].update_all(
                                fn.copy_u(k, 'msg'),
                                fn.mean('msg', f'm{k}'), etype='cite')
                            g['cited_by'].update_all(
                                fn.copy_u(k, 'msg'),
                                fn.mean('msg', f't{k}'), etype='cited_by')

                keys = list(g.ndata.keys())
                print(f'Involved label keys {keys}')
                for k in keys:
                    if k == tgt_type: continue
                    label_feats[k[:-1]] = g.ndata.pop(k)
                g = clear_hg(g, echo=False)

                condition = lambda ra,rb,rc,k: True
                check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid)

                # remove self effect on label feats
                mm_diag, mt_diag, tm_diag, tt_diag = torch.load(f'{args.dataset}_diag.pt')
                diag_values = {'mm': mm_diag, 'mt': mt_diag, 'tm': tm_diag, 'tt': tt_diag}
                for k in diag_values:
                    if k in label_feats:
                        label_feats[k] = label_feats[k] - diag_values[k].unsqueeze(-1) * label_onehot
                        assert torch.all(label_feats[k] > -1e-6)
                        print(k, torch.sum(label_feats[k] < 0), label_feats[k].min())

                condition = lambda ra,rb,rc,k: True
                check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid)

                label_emb = (label_feats['t'] + label_feats['tm'] + label_feats['mt'] + label_feats['tt']) / 4
                check_acc({'label_emb': label_emb}, condition, init_labels, train_nid, val_nid, test_nid)

            elif args.dataset == 'ogbn-mag':
                g.nodes['P'].data['P'] = label_onehot

                extra_metapath = [] # ['PAIAP']
                extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_label_hops + 1]

                print(f'Current num label hops = {args.num_label_hops}')
                if len(extra_metapath):
                    max_hops = max(args.num_label_hops + 1, max([len(ele) for ele in extra_metapath]))
                else:
                    max_hops = args.num_label_hops + 1

                g = hg_propagate(g, tgt_type, args.num_label_hops, max_hops, extra_metapath, echo=False)

                keys = list(g.nodes[tgt_type].data.keys())
                print(f'Involved label keys {keys}')
                for k in keys:
                    if k == tgt_type: continue
                    label_feats[k] = g.nodes[tgt_type].data.pop(k)
                g = clear_hg(g, echo=False)

                # label_feats = remove_self_effect_on_label_feats(label_feats, label_onehot)
                if self_value_dict is not None:
                    for k in label_feats.keys():
                        label_feats[k] = label_feats[k] - self_value_dict[k].unsqueeze(-1) * label_onehot
                else:
                    for k in ['PPP', 'PAP', 'PFP', 'PPPP', 'PAPP', 'PPAP', 'PFPP', 'PPFP']:
                        if k in label_feats:
                            diag = torch.load(f'{args.dataset}_{k}_diag.pt')
                            label_feats[k] = label_feats[k] - diag.unsqueeze(-1) * label_onehot
                            # assert torch.all(label_feats[k] > -1e-6)
                            print(k, torch.sum(label_feats[k] < 0), label_feats[k].min())

                condition = lambda ra,rb,rc,k: True
                check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid)

                if self_value_dict:
                    label_emb = 0
                    for k in label_feats.keys():
                        label_emb = label_emb + label_feats[k] / len(label_feats.keys())
                else:
                    label_emb = (label_feats['PPP'] + label_feats['PAP'] + label_feats['PP'] + label_feats['PFP']) / 4
                check_acc({'label_emb': label_emb}, condition, init_labels, train_nid, val_nid, test_nid)
        else:
            label_emb = torch.zeros((num_nodes, n_classes))

        label_feats = {k: v[init2sort] for k, v in label_feats.items()}

        label_feats = {k: v for k, v in label_feats.items() if k in archs[args.arch][1]}

        label_emb = label_emb[init2sort]


        if stage == 0:
            label_feats = {}
            label_path = []
        else:
            label_path = archs[args.arch][1]

        # =======
        # Eval loader
        # =======
        if stage > 0:
            del eval_loader
        eval_loader = []
        for batch_idx in range((num_nodes-trainval_point-1) // args.batch_size + 1):
            batch_start = batch_idx * args.batch_size + trainval_point
            batch_end = min(num_nodes, (batch_idx+1) * args.batch_size + trainval_point)

            batch_feats = {k: v[batch_start:batch_end] for k,v in feats.items()}
            batch_label_feats = {k: v[batch_start:batch_end] for k,v in label_feats.items()}
            batch_labels_emb = label_emb[batch_start:batch_end]
            eval_loader.append((batch_feats, batch_label_feats, batch_labels_emb))

        data_size = {k: v.size(-1) for k, v in feats.items()}



        # =======
        # Construct network
        # =======
        model = LMSPS(args.dataset,
            data_size, args.embed_size,
            args.hidden, n_classes,
            len(feats), len(label_feats), tgt_type,
            dropout=args.dropout,
            input_drop=args.input_drop,
            att_drop=args.att_drop,
            label_drop=args.label_drop,
            n_layers_2=args.n_layers_2,
            n_layers_3=args.n_layers_3,
            residual=args.residual,
            bns=args.bns, label_bns=args.label_bns,
             path=archs[args.arch][0],
            label_path=label_path,
            eps = args.eps,
            device = device
            )
        model = model.to(device)
        if stage == args.start_stage:
            print(model)
            print("# Params:", get_n_params(model))

        loss_fcn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)

        best_epoch = 0
        best_val_acc = 0
        best_test_acc = 0
        count = 0

        train_times = []

        for epoch in range(epochs):
            gc.collect()
            torch.cuda.empty_cache()
            start = time.time()
            if stage == 0:
                loss, acc = train(model, train_loader, loss_fcn, optimizer, evaluator, device, feats, label_feats, labels_cuda, label_emb, scalar=scalar)
            else:
                loss, acc = train_multi_stage(model, train_loader, enhance_loader, loss_fcn, optimizer, evaluator, device, feats, label_feats, labels_cuda, label_emb, predict_prob, args.gama, scalar=scalar)
            end = time.time()
            train_times.append(end - start)
            log = "Epoch {}, Time(s): {:.4f}, estimated train loss {:.4f}, acc {:.4f}\n".format(epoch, end-start, loss, acc*100)
            torch.cuda.empty_cache()

            if epoch % args.eval_every == 0:
                with torch.no_grad():
                    model.eval()
                    raw_preds = []

                    start = time.time()
                    for batch_feats, batch_label_feats, batch_labels_emb in eval_loader:
                        batch_feats = {k: v.to(device) for k,v in batch_feats.items()}
                        batch_label_feats = {k: v.to(device) for k,v in batch_label_feats.items()}
                        batch_labels_emb = batch_labels_emb.to(device)
                        raw_preds.append(model(batch_feats, batch_label_feats, batch_labels_emb).cpu())
                    raw_preds = torch.cat(raw_preds, dim=0)

                    loss_val = loss_fcn(raw_preds[:valid_node_nums], labels[trainval_point:valtest_point]).item()
                    loss_test = loss_fcn(raw_preds[valid_node_nums:valid_node_nums+test_node_nums], labels[valtest_point:total_num_nodes]).item()

                    preds = raw_preds.argmax(dim=-1)
                    val_acc = evaluator(preds[:valid_node_nums], labels[trainval_point:valtest_point])
                    test_acc = evaluator(preds[valid_node_nums:valid_node_nums+test_node_nums], labels[valtest_point:total_num_nodes])

                    end = time.time()
                    log += f'Evaluation Time: {end-start}, Val loss: {loss_val}, Test loss: {loss_test}\n'
                    log += 'Val acc: {:.4f}, Test acc: {:.4f}\n'.format(val_acc*100, test_acc*100)

                if val_acc > best_val_acc:
                    best_epoch = epoch
                    best_val_acc = val_acc
                    best_test_acc = test_acc

                    torch.save(model.state_dict(), f'{checkpt_file}_{stage}.pkl')
                    count = 0
                else:
                    count = count + args.eval_every
                    if count >= args.patience:
                        break
                log += "Best Epoch {},Val {:.4f}, Test {:.4f}".format(best_epoch, best_val_acc*100, best_test_acc*100)
            print(log, flush=True)

        print('average train times', sum(train_times) / len(train_times))
        print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(best_epoch, best_val_acc*100, best_test_acc*100))

        model.load_state_dict(torch.load(checkpt_file+f'_{stage}.pkl'))

        raw_preds = gen_output_torch(model, feats, label_feats, label_emb, all_loader, device)


    return [round(best_val_acc * 100, 2), round(best_test_acc * 100, 2)]


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='LMSPS')
    ## For environment costruction
    parser.add_argument("--seeds", nargs='+', type=int, default=[1],
                        help="the seed used in the training")
    parser.add_argument("--dataset", type=str, default="ogbn-mag")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--root", type=str, default='../data/')
    parser.add_argument("--stages", nargs='+',type=int, default=[200, 200, 200, 200, 200, 200],
                        help="The epoch setting for each stage.")
    ## For pre-processing
    parser.add_argument("--emb_path", type=str, default='../data/')
    parser.add_argument("--extra-embedding", type=str, default='',
                        help="the name of extra embeddings")
    parser.add_argument("--embed-size", type=int, default=256,
                        help="inital embedding size of nodes with no attributes")
    parser.add_argument("--num-hops", type=int, default=6,
                        help="number of hops for propagation of raw labels")
    parser.add_argument("--label-feats", action='store_true', default=False,
                        help="whether to use the label propagated features")
    parser.add_argument("--num-label-hops", type=int, default=4,
                        help="number of hops for propagation of raw features")
    ## For network structure
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout on activation")
    parser.add_argument("--n-layers-2", type=int, default=2,
                        help="number of layers of the downstream task")
    parser.add_argument("--n-layers-3", type=int, default=2,
                        help="number of layers of residual label connection")
    parser.add_argument("--input-drop", type=float, default=0.1,
                        help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.,
                        help="attention dropout of model")
    parser.add_argument("--label-drop", type=float, default=0.,
                        help="label feature dropout of model")
    parser.add_argument("--residual", action='store_true', default=False,
                        help="whether to connect the input features")
    parser.add_argument("--bns", action='store_true', default=False,
                        help="whether to process the input features")
    parser.add_argument("--label-bns", action='store_true', default=False,
                        help="whether to process the input label features")
    ## for training
    parser.add_argument("--amp", action='store_true', default=False,
                        help="whether to amp to accelerate training with float16(half) calculation")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=100,  # original 100
                        help="early stop of times of the experiment")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="the threshold of multi-stage learning, confident nodes "
                           + "whose score above this threshold would be added into the training set")
    parser.add_argument("--gama", type=float, default=5,
                        help="parameter for the KL loss")
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--reload", type=str, default='')
    parser.add_argument("--identity", action='store_true', default=False)
    parser.add_argument('--arch', type=str, default='DBLP')
    parser.add_argument("--eps", type=float, default=0)   #1e-12
    parser.add_argument("--edge_mask_ratio", type=float, default=0)

    parser.add_argument("--mask_seed", type=int, default=1)
    parser.add_argument("--max_mask_deg", type=int, default=None)
    parser.add_argument("--in_max_deg", type=int, default=None)
    parser.add_argument("--out_max_deg", type=int, default=None)

    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    assert args.dataset.startswith('ogbn')
    print(args)

    results = []
    for seed in args.seeds:
        args.seed = seed
        print('Restart with seed =', seed)
        result = main(args)
        results.append(result)
    print(results)
    
    results = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    results = np.array(results)
    val_res = results[:,0]
    test_res = results[:,1]

    print(f'val: {val_res}', f'test: {test_res}')
    print(f'val_mean: {np.mean(val_res):.2f}', f'val_std: {np.std(val_res):.2f}')
    print(f'test_mean: {np.mean(test_res):.2f}', f'test_std: {np.std(test_res):.2f}')