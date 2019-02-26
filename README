Discriminability as an Affinity Measure for Large-Scale Hierarchical Image Clustering
=====================================================================================

This reposority contains the experiments that I have conducted for
my master thesis at University of Passau.

To run this code you need to install the following dependencies:
- Python 3.6
- [PyTorch](https://pytorch.org/) >=1.0.0
- [faiss](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) >= 1.5.0
- scikit-learn >=0.20.0
- Jupyter Notebook

Further you need a copy of the [ImageNet](http://image-net.org/download) dataset, which must be
accessible at `datasets/imagenet-full`.

The experiments which depend on DeepCluster embeddings also require the download
of the DeepCluster pretrained model. To download the pretrained model simply execute:
```
cd deepcluster
download_model.sh
```

Discriminability vs. DeepCluster
--------------------------------
To run the first experiment series, which compares Discriminability against the
DeepCluster embeddings run the following notebooks:
- `create-imagenet-subsets.ipynb`
- `imagenet-subsets.ipynb`
- `imagenet-subsets-eval.ipynb`

Artificial Initial Clustering
-----------------------------
The second experiment, which explores the influence of noise and initial cluster
size on the quality of the affinity measure, can be executed with the following
notebooks:
- `imagenet-largescale.ipynb`
- `imagenet-largescale-eval.ipynb`

This experiment does not require the pretrained DeepCluster model.

Benefit of Retraining
---------------------
The third experiment, which explores whether their is a benefit from recomputing
the affinity measure by retraining the neural network after a merge decision can be reproduced
with the notebooks:
- `retraining.ipynb`
- `retraining-eval.ipynb`
