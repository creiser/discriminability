
import sys
import os
from shutil import move
import torch
import subprocess
from . import utils
import numpy as np
from deepcluster import deepcluster

def get_embeddings(hp):
    embedding_directory = 'results/1-embedding'
    utils.create_directory_if_not_exists(embedding_directory)
    
    embedding_filename = embedding_directory + '/{}'.format(hp['dataset']) + '.npy'
    
    if not os.path.isfile(embedding_filename):
        print('Generate embeddings for ' + hp['dataset'])
        args = {}
        args['verbose'] = True
        args['data'] = 'datasets/' + hp['dataset']
        args['arch'] = 'vgg16'
        args['sobel'] = True
        args['workers'] = 4
        args['resume'] = 'deepcluster/pretrained/vgg16/checkpoint.pth.tar'
        args['batch'] = 64
        args['embedding_filename'] = embedding_filename
        deepcluster.main(args)

    print('Loading embeddings from {}'.format(embedding_filename))
    embeddings = torch.from_numpy(np.load(embedding_filename))
    return embeddings.to(hp['device'])
