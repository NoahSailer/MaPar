#!/bin/bash
#
# Get the data and maps from the original (v1) LRGxPlanck
# paper
#
url=https://zenodo.org/record/5834378/files
#
rm -f data.tgz?download=1 LRGxPlanck_v1_data.tgz
rm -f maps.tgz?download=1 LRGxPlanck_v1_maps.tgz
#
wget -O LRGxPlanck_v1_data.tgz ${url}/data.tgz?download=1
#
wget -O LRGxPlanck_v1_maps.tgz ${url}/maps.tgz?download=1
#
