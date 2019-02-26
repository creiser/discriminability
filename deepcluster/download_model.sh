# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

MODELROOT="./pretrained"

mkdir -p ${MODELROOT}

for MODEL in vgg16 #alexnet
do
  mkdir -p "${MODELROOT}/${MODEL}"
  for FILE in checkpoint.pth.tar #model.caffemodel model.prototxt
  do
    wget -c "https://s3.amazonaws.com/deepcluster/${MODEL}/${FILE}" \
      -P "${MODELROOT}/${MODEL}" 

  done
done
