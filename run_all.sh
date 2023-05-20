#!/bin/bash
# ALGS = "Lin RLB RRLB"
for alg in Lin RLB RRLB
do
    python bid_V_update.py --alg $alg
    python bid_ss2.py --alg $alg
done