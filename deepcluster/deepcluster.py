# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pickle
import time
import sys

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from . import clustering
from . import models
from .util import AverageMeter, Logger, UnifLabelSampler


def main(args):
    # CNN
    if args['verbose']:
        print('Architecture: {}'.format(args['arch']))
    model = models.__dict__[args['arch']](sobel=args['sobel'])
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    if args['resume']:
        if os.path.isfile(args['resume']):
            print("=> loading checkpoint '{}'".format(args['resume']))
            checkpoint = torch.load(args['resume'])
            args['start_epoch'] = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            for key in list(checkpoint['state_dict'].keys()):
                if 'top_layer' in key:
                    del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args['resume'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args['resume']))

    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    # load the data
    end = time.time()
    dataset = datasets.ImageFolder(args['data'], transform=transforms.Compose(tra))
    if args['verbose']: print('Load dataset: {0:.2f} s'.format(time.time() - end))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args['batch'],
                                             num_workers=args['workers'],
                                             pin_memory=True)

    # remove head
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    # get the features for the whole dataset
    features = compute_features(dataloader, model, len(dataset), args)
    
    if 'embedding_filename' in args:
        print('Write DeepCluster features to', args['embedding_filename'])
        np.save(args['embedding_filename'], features)
    
    # cluster the features
    if 'clustering_filename' in args:
        # clustering algorithm to use
        cluster_algorithm = clustering.__dict__[args['clustering']](args['nmb_cluster'])
        clustering_loss = cluster_algorithm.cluster(features, verbose=args['verbose'])
        print('Write DeepCluster clustering to', args['clustering_filename'])
        np.save(args['clustering_filename'], deepcluster.images_lists)

def compute_features(dataloader, model, N, args):
    if args['verbose']:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    with torch.no_grad():
        for i, (input_tensor, _) in enumerate(dataloader):
            input_var = input_tensor.cuda()
            aux = model(input_var).data.cpu().numpy()

            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32')

            if i < len(dataloader) - 1:
                features[i * args['batch']: (i + 1) * args['batch']] = aux.astype('float32')
            else:
                # special treatment for final batch
                features[i * args['batch']:] = aux.astype('float32')

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args['verbose'] and (i % 5) == 0:
                print('{0} / {1}\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                      .format(i, len(dataloader), batch_time=batch_time))
    return features
    
if __name__ == '__main__':
    main()
