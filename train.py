'''
train.py
Author: Kaushal Paneri
Project: DCRNN
Date of Creation: 12/29/18
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import torch
import yaml

from model.dcrnn_model import DCRNNSupervisor
from lib.utils import load_graph_data
def main(args):

    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        #print(sensor_ids)
        #print(sensor_id_to_ind)
        #print(adj_mx.shape)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(device)

        supervisor = DCRNNSupervisor(adj_mtx=adj_mx, **supervisor_config)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/dcrnn_config.yaml', type=str,
                        help='Configuration Filename for restoring model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)




