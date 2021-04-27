#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

import gzip

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--workspace_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=10000, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--record', action='store_true')
    parser.add_argument('--topk', default=100, type=int)
    
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    
def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, t, r = line.strip().split('\t')
            triples.append((int(h), int(r), int(t)))
    return triples

def read_triple_fb(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
        
def main(args):
    # if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        # raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)
    
    # with open(os.path.join(args.data_path, 'entities.dict')) as fin:
    #     entity2id = dict()
    #     id2entity = dict()
    #     for line in fin:
    #         eid, entity = line.strip().split('\t')
    #         entity2id[entity] = int(eid)
    #         id2entity[int(eid)] = entity

    # with open(os.path.join(args.data_path, 'relations.dict')) as fin:
    #     relation2id = dict()
    #     id2relation = dict()
    #     for line in fin:
    #         rid, relation = line.strip().split('\t')
    #         relation2id[relation] = int(rid)
    #         id2relation[int(rid)] = relation
    
    # # Read regions for Countries S* datasets
    # if args.countries:
    #     regions = list()
    #     with open(os.path.join(args.data_path, 'regions.list')) as fin:
    #         for line in fin:
    #             region = line.strip()
    #             regions.append(entity2id[region])
    #     args.regions = regions

    '''amazon dataset'''
    with open(os.path.join(args.data_path, 'entity2id.txt')) as fin:
        entity2id = dict()
        id2entity = dict()
        for line in fin:
            if len(line.strip().split('\t')) < 2:
                continue
            entity, eid = line.strip().split('\t')
            entity2id[entity] = int(eid)
            id2entity[int(eid)] = entity

    with open(os.path.join(args.data_path, 'relation2id.txt')) as fin:
        relation2id = dict()
        id2relation = dict()
        for line in fin:
            if len(line.strip().split('\t')) < 2:
                continue
            relation, rid = line.strip().split('\t')
            relation2id[relation] = int(rid)
            id2relation[int(rid)] = relation

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    # --------------------------------------------------
    # Comments by Meng:
    # During training, pLogicNet will augment the training triplets,
    # so here we load both the augmented triplets (train.txt) for training and
    # the original triplets (train_kge.txt) for evaluation.
    # Also, the hidden triplets (hidden.txt) are also loaded for annotation.
    # --------------------------------------------------
    # train_triples = read_triple(os.path.join(args.workspace_path, 'train_kge.txt'), entity2id, relation2id)
    # logging.info('#train: %d' % len(train_triples))
    # train_original_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    # logging.info('#train original: %d' % len(train_original_triples))
    # valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    # logging.info('#valid: %d' % len(valid_triples))
    # test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    # logging.info('#test: %d' % len(test_triples))
    # hidden_triples = read_triple(os.path.join(args.workspace_path, 'hidden.txt'), entity2id, relation2id)
    # logging.info('#hidden: %d' % len(hidden_triples))

    train_triples = read_triple(os.path.join(args.workspace_path, 'train_kge.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    train_original_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train original: %d' % len(train_original_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'kg_val_triples_Cell_Phones_and_Accessories.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'kg_test_triples_Cell_Phones_and_Accessories.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    test_candidates = np.load(os.path.join(args.data_path, 'rec_test_candidate100.npz'))['candidates'][:, 1:]
    # test_candidates = np.load('/common/users/yz956/kg/code/OpenDialKG/cand.npy')
    # hidden_triples = read_triple(os.path.join(args.workspace_path, 'hidden.txt'), entity2id, relation2id)
    hidden_triples = read_triple("/common/users/yz956/kg/code/KBRD/data/cpa/cpa/hidden10_cpa.txt", entity2id, relation2id)
    logging.info('#hidden: %d' % len(hidden_triples))
    
    #All true triples
    all_true_triples = train_original_triples + valid_triples + test_triples
    
    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()
    
    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    if args.do_train:
        logging.info('learning_rate = %f' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('test_batch_size = %d' % args.test_batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    if args.record:
        local_path = args.workspace_path
        ensure_dir(local_path)

        opt = vars(args)
        with open(local_path + '/opt.txt', 'w') as fo:
            for key, val in opt.items():
                fo.write('{} {}\n'.format(key, val))
    
    # Set valid dataloader as it would be evaluated during training
    
    if args.do_train:
        training_logs = []
        
        #Training Loop
        for step in range(init_step, args.max_steps):
            
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            
            training_logs.append(log)
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                # optimizer = torch.optim.Adam(
                #     filter(lambda p: p.requires_grad, kge_model.parameters()), 
                #     lr=current_learning_rate
                # )
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_learning_rate
                warm_up_steps = warm_up_steps * 3
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if args.do_valid and (step + 1) % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics, preds = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)
        
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics, preds = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)
        
        # --------------------------------------------------
        # Comments by Meng:
        # Save the prediction results of KGE on validation set.
        # --------------------------------------------------

        if args.record:
            # Save the final results
            with open(local_path + '/result_kge_valid.txt', 'w') as fo:
                for metric in metrics:
                    fo.write('{} : {}\n'.format(metric, metrics[metric]))

            # Save the predictions on test data
            with open(local_path + '/pred_kge_valid.txt', 'w') as fo:
                for h, r, t, f, rk, l in preds:
                    fo.write('{}\t{}\t{}\t{}\t{}\n'.format(id2entity[h], id2relation[r], id2entity[t], f, rk))
                    for e, val in l:
                        fo.write('{}:{:.4f} '.format(id2entity[e], val))
                    fo.write('\n')
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics, preds = kge_model.test_step(kge_model, test_triples, test_candidates, all_true_triples, args)
        log_metrics('Test', step, metrics)
        
        # --------------------------------------------------
        # Comments by Meng:
        # Save the prediction results of KGE on test set.
        # --------------------------------------------------

        if args.record:
            # Save the final results
            with open(local_path + '/result_kge.txt', 'w') as fo:
                for metric in metrics:
                    fo.write('{} : {}\n'.format(metric, metrics[metric]))

            # Save the predictions on test data
            with open(local_path + '/pred_kge.txt', 'w') as fo:
                for h, r, t, f, rk, l in preds:
                    # fo.write('{}\t{}\t{}\t{}\t{}\n'.format(id2entity[h], id2relation[r], id2entity[t], f, rk))
                    fo.write('{}\t{}\t{}\t{}\t{}\n'.format(str(h), str(t), str(r), f, rk))
                    for e, val in l:
                        fo.write('{}:{:.4f} '.format(id2entity[e], val))
                    fo.write('\n')

    # --------------------------------------------------
    # Comments by Meng:
    # Save the annotations on hidden triplets.
    # --------------------------------------------------

    if args.record:
        # Annotate hidden triplets
        ann, train = [], []
        d = {}
        with open('../../sample_pre.txt') as ft:
            for line in ft:
                line = line.strip().split('\t')
                train.append(line[1:])
        for u in range(61254):
            hiddens = []
            for i in range(61254, 108858):
                hiddens.append((u, 0, i))
            scores = kge_model.infer_step(kge_model, hiddens, args)
            score_np = np.array(scores)
            d = dict(zip(range(61254, 108858), scores))
            d = sorted(d.items(), key=lambda x: x[1], reverse=True)
            
            d_50 = d[:50]
            for idx, t in enumerate(train[u]):
                for (tt, prob) in d_50:
                    if int(t) == tt:
                        d_50.remove((tt, prob))
                        d_50.append(d[50 + idx])
            assert len(d_50) == 50
            d = {}

            d_50 = d
            ann.append(d_50)
        with open(local_path + '/annotation.txt', 'w') as fo:
            for idx, d in enumerate(ann):
                for (t, score) in d:
                    fo.write(str(idx) + '\t' + str(t) + '\t0\t' + str(score) + '\n')
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics, preds = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
        
if __name__ == '__main__':
    main(parse_args())
