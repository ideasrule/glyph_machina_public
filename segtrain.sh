#!/bin/bash
ls train_data/*.xml > seg_train_list.txt
ls test_data/*.xml > seg_test_list.txt

ketos --workers 16 -d "cuda:0" --deterministic segtrain --augment --suppress-regions -t seg_train_list.txt -e seg_test_list.txt -N 50 -f xml -bl -i pretrained_backbone_blla.mlmodel --resize new
