from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import os 
sys.path.append(os.getcwd())
import collections
import copy
import time
from datetime import timedelta
import json
from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from spcl import datasets
from spcl import models
from spcl.models.hm import HybridMemory, ClusterMemory, ClusterMemory2
from spcl.trainers import SpCLTrainer_USL,VDGTrainer_USL, VDGTrainer_USL_view
# from spcl.evaluators import Evaluator, extract_features
from spcl.evaluators import Evaluator, extract_features, extract_aug_features, extract_features_view
from spcl.utils.data import IterLoader
from spcl.utils.data import transforms as T
from spcl.utils.data.sampler import RandomMultipleGallerySampler
# from spcl.utils.data.preprocessor import Preprocessor
from spcl.utils.data.preprocessor import Preprocessor,Preprocessor_aug, Preprocessor_aug2, Preprocessor2
from spcl.utils.logging import Logger
from spcl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from spcl.utils.faiss_rerank import compute_jaccard_distance

start_epoch = best_mAP = 0

def get_data(name, data_dir):
    root = osp.join(data_dir, "market1501")
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                             std= [0.5, 0.5, 0.5])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor_aug2(train_set, root=dataset.images_dir, transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    return train_loader



def get_train_augloader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None, selected_set = None):

    normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                             std= [0.5, 0.5, 0.5])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor_aug2(train_set, root=dataset.images_dir, transform=train_transformer, selected_list=selected_set),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None): 
    normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                             std= [0.5, 0.5, 0.5])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader



def get_cluster_loader(dataset, height, width, batch_size, workers, testset=None): 
    normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                             std= [0.5, 0.5, 0.5])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor2(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader
def get_augset_loader(dataset, height, width, batch_size, workers, augpath=None):
    normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                             std= [0.5, 0.5, 0.5])
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    augset = []
    for img in os.listdir(augpath):
        augset.append((os.path.join(augpath, img), img,  -1))  #camid,
    # print(len(augset),len( dataset.train))
    test_loader = DataLoader(
        Preprocessor(augset, root=None, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return test_loader

def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0)
    # use CUDA
    model.cuda()
    weights = torch.load("./pretrained/iteration_200000.pt")['state_dict']
    # print(type(weights['state_dict'])) #.keys()
    body_dict = collections.OrderedDict()
    for key in weights.keys():
        if 'body' in key:
            body_dict["base."+key[18:]]=weights[key]
    model = models.create('resnet50', num_features=0, norm=True, dropout=0, num_classes=0)
    # use CUDA
    model.cuda()
    print(len(model.state_dict().keys()))
    print()
    # model.load_state_dict(body_dict)
    # model = nn.DataParallel(model)
    model_key =list( model.state_dict().keys())
    body_key = list( body_dict.keys())
    for i in range(len(body_dict.keys())):
        model.state_dict()[model_key[i]].copy_(body_dict[body_key[i]])
    model = nn.DataParallel(model)
    # model.load_state_dict(torch.load("/home/lzy/VDG/SpCL/logs/spcl_usl/baseline_0.5/model_best.pth.tar")["state_dict"])
    return model

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters>0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, 128, args.workers) #args.batch_size

    # Create model
    model = create_model(args)
    # Create hybrid memory
    # memory = HybridMemory(model.module.num_features, len(dataset.train),
    #                         temp=args.temp, momentum=args.momentum).cuda()
    memory = ClusterMemory2(10, 2048)
    # Initialize target-domain instance features
    print("==> Initialize instance features in the hybrid memory")
    cluster_loader = get_cluster_loader(dataset, args.height, args.width,
                                    args.batch_size, args.workers, testset=sorted(dataset.train))
    selection_th = args.reliability
    lambda_v = args.lambda_view
    start_epoch = args.start_epoch
    # features, _ = extract_features(model, cluster_loader, print_freq=50)
    # features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
    # # memory.features = F.normalize(features, dim=1).cuda()
    # # del cluster_loader, features
    # del features

    # Evaluator
    evaluator = Evaluator(model)
    aug_loader = get_augset_loader(dataset,  args.height, args.width, 128, args.workers,"./examples/data/market_train_fpn_final")
    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = VDGTrainer_USL_view(model, memory, lambda_v, start_epoch) #SpCLTrainer_USL(model, memory)
    mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
    for epoch in range(args.epochs):
        # Calculate distance
        print('==> Create pseudo labels for unlabeled data with self-paced policy')
        features_aug, _ ,_ = extract_aug_features(model, aug_loader,print_freq=50)

        # features = memory.features.clone()
        features, _, views = extract_features_view(model, cluster_loader, print_freq=50)
        views = [views[f] for f, _, _,_ in sorted(dataset.train)]
        views = np.array(views)
        select_augs = collections.defaultdict(list)
        select_augs_2 = collections.defaultdict(list)

        for f, _,_,_ in sorted(dataset.train):
                f_augfeatures = []
                # # print(f)
                # key = f.split("/")[-1][:-4]
                # if key not in view_point_id.keys():
                #     # print(key,"+++++++")
                #     # continue
                #     features[f] = torch.cat([features[f], features[f]])
                #     continue
                # for aug_img in view_point_id[key]:
                for key in range(4): 
                    aug_feature = features_aug[f.split("/")[-1][:-4]+"_view"+str(key)+".jpg"]
                    scores = torch.matmul(features[f], aug_feature.T)
                    select_augs_2[f].append(f.split("/")[-1][:-4]+"_view"+str(key)+".jpg"+str(scores))
                    if scores>selection_th:
                        select_augs[f].append(f.split("/")[-1][:-4]+"_view"+str(key)+".jpg")
                    f_augfeatures.append(features_aug[f.split("/")[-1][:-4]+"_view"+str(key)+".jpg"])
                # # f_augfeatures.append(features[f])
                # f_features = torch.stack(f_augfeatures)
                # scores = torch.matmul(features[f], f_features.T)
                
                # # print(scores)
                # index_select = scores>0.3
                # if sum(index_select)==0:
                #     features[f] = torch.cat([features[f], features[f]])
                #     continue

                # scores =torch.softmax(scores[index_select], dim=0)
                # aug_meanfeature = torch.sum(scores.unsqueeze(-1)*f_features[index_select], dim=0)
                # # print(torch.matmul(features[f], f_features.T))
                # # aug_meanfeature = torch.stack(f_augfeatures, dim=0).mean(0)
                # features[f] = torch.cat([features[f], aug_meanfeature])
        # json.
        f = open("./logs/"+str(epoch)+".json","w")
        json.dump(select_augs_2, f)
        f.close()
        features = torch.cat([features[f].unsqueeze(0) for f, _, _ ,_ in sorted(dataset.train)], 0)
        rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)
        # del features
        if (epoch==0):
            # DBSCAN cluster
            eps = args.eps
            eps_tight = eps-args.eps_gap
            eps_loose = eps+args.eps_gap
            print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_tight, eps_loose))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_tight = DBSCAN(eps=eps_tight, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_loose = DBSCAN(eps=eps_loose, min_samples=4, metric='precomputed', n_jobs=-1)
        elif (epoch==50):
            # DBSCAN cluster
            eps = 0.5
            eps_tight = eps-args.eps_gap
            eps_loose = eps+args.eps_gap
            print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_tight, eps_loose))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_tight = DBSCAN(eps=eps_tight, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_loose = DBSCAN(eps=eps_loose, min_samples=4, metric='precomputed', n_jobs=-1)
        
        # select & cluster images as training set of this epochs
        pseudo_labels = cluster.fit_predict(rerank_dist)
        pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist)
        pseudo_labels_loose = cluster_loose.fit_predict(rerank_dist)
        num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)
        num_ids_loose = len(set(pseudo_labels_loose)) - (1 if -1 in pseudo_labels_loose else 0)

        # generate new dataset and calculate cluster centers
        def generate_pseudo_labels(cluster_id, num):
            labels = []
            outliers = 0
            for i, ((fname, _, _, cid), id) in enumerate(zip(sorted(dataset.train), cluster_id)):
                if id!=-1:
                    labels.append(id)
                else:
                    # labels.append(num+outliers)
                    # outliers += 1
                    labels.append(-1)
            return torch.Tensor(labels).long()

        pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)
        pseudo_labels_tight = generate_pseudo_labels(pseudo_labels_tight, num_ids_tight)
        pseudo_labels_loose = generate_pseudo_labels(pseudo_labels_loose, num_ids_loose)
        # compute R_indep and R_comp
        N = pseudo_labels.size(0)
        label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
        label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
        label_sim_loose = pseudo_labels_loose.expand(N, N).eq(pseudo_labels_loose.expand(N, N).t()).float()
        R_comp = 1-torch.min(label_sim, label_sim_tight).sum(-1)/torch.max(label_sim, label_sim_tight).sum(-1)
        R_indep = 1-torch.min(label_sim, label_sim_loose).sum(-1)/torch.max(label_sim, label_sim_loose).sum(-1)
        assert((R_comp.min()>=0) and (R_comp.max()<=1))
        assert((R_indep.min()>=0) and (R_indep.max()<=1))
        cluster_R_comp, cluster_R_indep = collections.defaultdict(list), collections.defaultdict(list)
        cluster_img_num = collections.defaultdict(int)
        for i, (comp, indep, label) in enumerate(zip(R_comp, R_indep, pseudo_labels)):
            cluster_R_comp[label.item()].append(comp.item())
            cluster_R_indep[label.item()].append(indep.item())
            cluster_img_num[label.item()]+=1

        cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
        cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
        cluster_R_indep_noins = [iou for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys())) if cluster_img_num[num]>1]
        if (epoch==0):
            indep_thres = np.sort(cluster_R_indep_noins)[min(len(cluster_R_indep_noins)-1,np.round(len(cluster_R_indep_noins)*0.9).astype('int'))]

        pseudo_labeled_dataset = []
        outliers = 0
        # for i, label in enumerate(pseudo_labels):
        #     indep_score = cluster_R_indep[label.item()]
        #     comp_score = R_comp[i]
        #     if ((indep_score<=indep_thres) and (comp_score.item()<=cluster_R_comp[label.item()])):
        #         # pseudo_labeled_dataset.append((fname,label.item(),cid))
        #         continue
        #     else:
        #         pseudo_labels[i] = -1

        pseudo_labels = pseudo_labels.numpy()
        # pid2label = {pid: label for label, pid in enumerate(set(pseudo_labels))}
        pid2label = {pid: label for label, pid in enumerate(set(pseudo_labels[pseudo_labels>-1]))}
        for i in range(len(pseudo_labels)):
            if pseudo_labels[i] != -1:
                pseudo_labels[i] = pid2label[pseudo_labels[i]]
        for i, ((fname, _, vid, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label>-1:
                pseudo_labeled_dataset.append((fname,label,vid, cid))
            # indep_score = cluster_R_indep[label.item()]
            # comp_score = R_comp[i]
            # if ((indep_score<=indep_thres) and (comp_score.item()<=cluster_R_comp[label.item()])):
            #     pseudo_labeled_dataset.append((fname,label.item(),cid))
            # else:
            #     pseudo_labeled_dataset.append((fname,len(cluster_R_indep)+outliers,cid))
            #     pseudo_labels[i] = len(cluster_R_indep)+outliers
            #     outliers+=1
        @torch.no_grad()
        def generate_cluster_features(labels, features,views):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]
            centers = torch.stack(centers, dim=0)
            perview_memory = []
            concate_intra_class = []
            init_intra_id_feat=[]
            view_class_mapper = []
            views = np.array(views)
            labels = np.array(labels)
            for vv in np.unique(views): #torch.unique(views): #
                percam_ind = np.where(views == vv)[0] #torch.nonzero(views == vv).squeeze(-1)
                percam_feature = features[percam_ind].numpy()
                uniq_class = np.unique(labels[percam_ind])
                uniq_class = uniq_class[uniq_class >= 0]
                percam_id_feature = np.zeros((len(uniq_class), features.shape[1]), dtype=np.float32)
                cnt = 0
                for lbl in np.unique(uniq_class):
                    if lbl >= 0:
                        ind = np.where(labels[percam_ind] == lbl)[0]
                        id_feat = np.mean(percam_feature[ind], axis=0)
                        percam_id_feature[cnt, :] = id_feat
                        # intra_id_labels.append(lbl)
                        cnt += 1
                percam_id_feature = percam_id_feature / np.linalg.norm(percam_id_feature, axis=1, keepdims=True)
                init_intra_id_feat.append(torch.from_numpy(percam_id_feature))
                concate_intra_class.append(torch.from_numpy(uniq_class))
                cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
                view_class_mapper.append(cls_mapper)

                # cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
                # memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera
                if len(init_intra_id_feat) > 0:
                    # print('initializing ID memory from updated embedding features...')
                    proto_memory = init_intra_id_feat[vv]
                    proto_memory = proto_memory.cuda()
                    perview_memory.append(proto_memory.detach())
            return centers, perview_memory, concate_intra_class, view_class_mapper

        centroids = collections.defaultdict(list)
    
        for i, label in enumerate(pseudo_labels):
            if label==-1:
                continue
            centroids[pseudo_labels[i]].append(features[i])
        centroids = [
            torch.stack(centroids[idx], dim=0).mean(0) for idx in sorted(centroids.keys())
        ]
        centroids = torch.stack(centroids, dim=0)
    
        centroids, perview_memory, concate_intra_class, view_class_mapper = generate_cluster_features(pseudo_labels, features,views)
        trainer.view_proxy = perview_memory
        trainer.view_classes = concate_intra_class
        trainer.view_label_mapper = view_class_mapper
        trainer.views_label = views
        concate_intra_class = torch.cat(concate_intra_class)
        concate_intra_class = concate_intra_class.cuda()
        percam_tempV = []
        for vv in np.unique(views):
            percam_tempV.append(perview_memory[vv].detach().clone())
        percam_tempV= torch.cat(percam_tempV, dim=0).cuda()
        # concate_intra_class = []
        # percam_tempV = []
        # del cluster_loader, features

        view_centroids = []
        view_centroid_labels = []


        # views = np.array(views)
        # pseudo_labels = np.array(pseudo_labels)
        # for vv in np.unique(views):
        #     view_index = np.where(views == vv)[0]
        #     view_features = features[view_index].numpy()
        #     view_class_label = np.unique(pseudo_labels[view_index])
        #     view_class_label = view_class_label[view_class_label>=0]
        #     view_controids_feature = np.zeros((centroids.shape[0], features.shape[1]), dtype=np.float32)
        #     count =  0
        #     for view_label in view_class_label:
        #         if view_label>=0:
        #             index = np.where(pseudo_labels[view_index]==view_label)[0]
        #             view_point_feat = np.mean(view_features[index], axis =0)
        #             view_controids_feature[view_label, :] = view_point_feat
        #             # intra_id_labels.append(lbl)
        #             # count += 1
        #     view_controids_feature = view_controids_feature / np.linalg.norm(view_controids_feature, axis=1, keepdims=True)
        #     view_centroids.append(torch.from_numpy(view_controids_feature))
            # view_centroid_labels.append(torch.from_numpy(view_class_label))

        # view_centroid_labels = torch.cat(view_centroid_labels)
        # view_centroid_labels = view_centroid_labels.cuda()

        # statistics of clusters and un-clustered instances

        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label.item()]+=1
        index2label = np.fromiter(index2label.values(), dtype=float)
        # print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances, R_indep threshold is {}'
        #             .format(epoch, (index2label>1).sum(), (index2label==1).sum(), 1-indep_thres))

        # memory.num_samples = len(pseudo_labels[pseudo_labels>-1])

        memory.features = F.normalize(centroids, dim=1).cuda()

        memory.labels =  torch.Tensor(pseudo_labels[pseudo_labels>-1]).long()
        trainer.features = F.normalize(centroids, dim=1).cuda()
        trainer.labels =  torch.Tensor(pseudo_labels[pseudo_labels>-1]).long()
        # model.module.classifier = nn.Linear(2048, centroids.shape[0], bias=False).cuda()
        # model.module.classifier.weight.data.copy_(
        #     F.normalize(centroids[:, :2048], dim=1).float().cuda())
        # if epoch in [20,40]:

        #     args.lr = args.lr/10
        # params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
        # optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        print(pseudo_labels.shape, centroids.shape,len(pseudo_labels[pseudo_labels>-1]))
        train_loader = get_train_augloader(args, dataset, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset, selected_set = select_augs)
        train_loader.new_epoch()
        trainer.train(epoch, train_loader, optimizer,
                    print_freq=args.print_freq, train_iters=len(train_loader),percam_tempV = percam_tempV,  concate_intra_class = concate_intra_class)

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP>best_mAP)   
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
        lr_scheduler.step()

    print ('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--reliability', type=float, default=0.8,
                        help="sample selection threshold")
    parser.add_argument('--lambda_view', type=float, default=0.1,
                        help="view-aware contrastive loss weights")
    parser.add_argument('--start_epoch', type=int, default=0, help="viewpoint-aware contrastive loss start epoch")
                        
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()