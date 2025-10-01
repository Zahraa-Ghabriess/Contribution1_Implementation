import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 8,
        help = 'batch size when trained on client'
    )
    parser.add_argument(
        '--num_communication',
        type = int,
        default=1,
        help = 'number of communication rounds with the cloud server'
    )
    parser.add_argument(
        '--num_local_update',
        type=int,
        default=1,
        help='number of local update (tau_1)'
    )
    parser.add_argument(
        '--num_edge_aggregation',
        type = int,
        default=1,
        help = 'number of edge aggregation (tau_2)'
    )
    parser.add_argument(
        '--verbose',
        type = int,
        default = 0,
        help = 'verbose for print progress bar'
    )
    parser.add_argument(
        '--frac',
        type = float,
        default = 1,
        help = 'fraction of participated clients'
    )
    parser.add_argument(
        '--num_clients',
        type = int,
        default = 10,
        help = 'number of all available clients'
    )
    parser.add_argument(
        '--num_edges',
        type = int,
        default= 1,
        help= 'number of edges'
    )
    parser.add_argument(
        '--seed',
        type = int,
        default = 1,
        help = 'random seed (defaul: 1)'
    )
    parser.add_argument(
        '--show_dis',
        type= int,
        default= 0,
        help='whether to show distribution'
    )
    parser.add_argument(
        '--local_model',
        default=0,
        type=int
    )


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
