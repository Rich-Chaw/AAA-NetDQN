#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from GraphDQN import GraphDQN

def main():
    dqn = GraphDQN(
        g_type = 'barabasi_albert',
        gnn_model = 'GIN',
        target_graph = "Digg",
        num_min = 30,
        num_max = 50,
        ckpt_file = None
    )
    dqn.Train()


if __name__=="__main__":
    main()
