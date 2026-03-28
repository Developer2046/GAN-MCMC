from typing import Tuple
import os
from os import path as pt

import numpy as np
import torch

from hyperparameters import SIGCWGAN_CONFIGS
from lib import ALGOS
from lib.algos.base import BaseConfig
from lib.data import get_data, get_data_mcmc
from lib.utils import pickle_it, load_pickle
#from lib.utils import sample_indices
import matplotlib.pyplot as plt


def get_algo_config(dataset, data_params):
    """ Get the algorithms parameters. """
    key = dataset
    if dataset == 'VAR':
        key += str(data_params['dim'])
    elif dataset == 'STOCKS':
        key += '_' + '_'.join(data_params['assets'])
    if key in SIGCWGAN_CONFIGS.keys():
        return SIGCWGAN_CONFIGS[key]
    elif dataset[:3] == 'ett':
        return SIGCWGAN_CONFIGS['ETT']
    else:
        return SIGCWGAN_CONFIGS['Other']


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_algo(algo_id, base_config, dataset, data_params, x_real):
    if algo_id == 'SigCWGAN':
        algo_config = get_algo_config(dataset, data_params)
        algo = ALGOS[algo_id](x_real=x_real, config=algo_config, base_config=base_config)
    else:
        algo = ALGOS[algo_id](x_real=x_real, base_config=base_config)
    return algo


def discretization(samplers, bins):
    freq, bin_edges = np.histogramdd(samplers, bins)
    n = len(samplers)
    freq_rate = freq / n
    # calculate the bin width along each dimension
    # since the width of each bin is equivalent with each other
    # we just calculate the first bin width
    bin_w_list = list()
    for ix in range(len(bin_edges)):
        bin_w = bin_edges[ix][1] - bin_edges[ix][0]
        bin_w_list.append(bin_w)
        
    return freq_rate, bin_edges, bin_w_list

def get_joint_prob(freq_rate, bin_edges, vector):
    # calculate the index along the different axis
    index_list = list()
    for ix in range(len(vector)):
        #print("bin edges : ", np.array(bin_edges).shape)
        
        index = int((vector[ix] - bin_edges[ix][0])/(bin_edges[ix][1] - bin_edges[ix][0]))
        index_list.append(index)
        
    # if the vector is out of the range, return a small random value or zero instead
    # return a small value, since it is discrete system and our distribution may not be sufficient
    if not all(0 <= ix < len(freq_rate) for ix in index_list):
        return 1e-30 * np.random.rand()
    else:
        return freq_rate[tuple(index_list)] + 1e-30 * np.random.rand()


def run_mcmc(algo_id, base_config, base_dir, dataset, spec, data_params={}):
    """ Create the experiment directory, calibrate algorithm, store relevant parameters. """
    print('Executing: %s, %s, %s' % (algo_id, dataset, spec))

    # make dirs
    experiment_directory = pt.join(base_dir, dataset, spec, 'seed={}'.format(base_config.seed), algo_id)
    base_config.experiment_directory = experiment_directory

    # set seeds
    # set_seed(base_config.seed)

    # obtain data
    pipeline, real_data, real_block, condition_block  = get_data_mcmc(dataset, base_config.p, base_config.q, **data_params)
    
    real_data = real_data.squeeze(0).cpu().detach().numpy()
    #print("Shape Real Data : ", real_data.shape)
    
    # set algo
    algo = get_algo(algo_id, base_config, dataset, data_params, real_block)

    # load model 
    G_weight = load_pickle(os.path.join(experiment_directory, 'G_weights_{}_{}_{}.torch'.format(args.algos, dataset, args.weight_index)))
    algo.G.load_state_dict(G_weight)
    
    beta = 1
    gamma = 1
    
    print("Beta : ", beta)
    # the following is the process of mcmc
    diff_all = np.diff(real_data, axis=0)
    
    old_data = np.array(real_data[base_config.p] - real_data[base_config.p-1])
    time_series = list()
    freq_rate, bin_edges, bin_w_list = discretization(diff_all, bins=4)
    
    
    steps = 32   
    length = len(real_data) - base_config.p 

    # time stamp for the data
    # we need the time stamp to track the data we generated
    time_range = np.linspace(0.05, 0.95, len(real_data))  # 5000 points from 0 to 1 inclusive

    ix = 0
    while( ix < length - 1):

        ix = len(time_series)
        # selected the specific time point 
        time_point = time_range[ix + base_config.p]
        # print(time_point)
        # retrieve the data block for sampling
        
        if ix < steps:
            start = 0
            end   = start + ix + 1
            print(start, end)

        elif ix >= steps:
            start = (int(ix/steps) - 1) * steps + int(ix%steps) + 1
            end = start + steps
            print(start, end)

        elif ix + steps >= length:
            start = (int(ix/steps) - 1) * steps + int(ix%steps) + 1
            end = length
            print(start, end)
            
        real_batch = real_block[start:end].to(base_config.device)
        cond_batch = condition_block[start:end].to(base_config.device)

        cur_batch_size, time_steps, width = real_batch.shape

        sample_value = np.zeros(width)
       
        counter = 0
        record_list = list()
        while(len(time_series)!= (ix+1)):
            
            x_past = real_batch[:, :base_config.p, :].clone().to(base_config.device)
            
            x_fake_future = algo.G.sample(steps, x_past)
            
            if isinstance(x_fake_future, Tuple):
                x_fake = x_fake_future[0]
            else:
                x_fake = x_fake_future
            
            # take out the value for the corresponding time stamp
            mask = (cond_batch == time_point)
            batch_idx, time_idx, _ = mask.nonzero(as_tuple=True)
            

            gen_value = x_fake[batch_idx, time_idx, :].squeeze(1).detach().cpu().numpy()

            counter = counter + 1
            
            for index in range(len(gen_value)):
                sample_value = gen_value[index]
                # print("sample Value : ", sample_value)
                record_list.append(sample_value)

            if len(time_series) != 0:
                sampled_data = beta * np.array(sample_value - time_series[-1]) + (1 - beta) * \
                                np.array(sample_value - real_data[ix + base_config.p])
            else:
                sampled_data = np.array(sample_value - real_data[base_config.p])

            alpha = np.min([get_joint_prob(freq_rate, bin_edges, sampled_data)/get_joint_prob(freq_rate, bin_edges, old_data),\
                            1])
            
            if np.random.rand() < alpha:
                if len(time_series) == 0:
                    final_value = sample_value
                    old_data = sampled_data
                else:
                    final_value = gamma * sample_value + (1 - gamma) * np.array(time_series[-1])
                    old_data = sampled_data
                time_series.append(final_value)
                
                print("add one data piece into the series", final_value, "Index :", len(time_series))
                break
            #else:
            #    old_data = sampled_data
            
            # add average to the list, since it is possible the model is 
            # not well train and it needs long time to converge
            if counter > 300:
                average_sample = np.mean(record_list, axis=0, keepdims=True)[0]
                time_series.append(average_sample)
                print("Add the average samples into the series", average_sample, "Index :", len(time_series))
                
                
    np.save(os.path.join(experiment_directory, 'mcmc_gen_00_{}_{}_{}.npy'.format(args.algos, dataset, args.index)), time_series)
    print('complete generation')
                
        
def run_test(algo_id, base_config, base_dir, dataset, spec, data_params={}):
    """ Create the experiment directory, calibrate algorithm, store relevant parameters. """
    print('Executing: %s, %s, %s' % (algo_id, dataset, spec))

    # make dirs
    experiment_directory = pt.join(base_dir, dataset, spec, 'seed={}'.format(base_config.seed), algo_id)
    base_config.experiment_directory = experiment_directory

    # set seeds
    # set_seed(base_config.seed)

    # obtain data
    # pipeline, x_real_train 
    pipeline, x_real, x_real_block = get_data(dataset, base_config.p, base_config.q, **data_params)
    

    # set algo
    algo = get_algo(algo_id, base_config, dataset, data_params, x_real_block)

    # load model 
    G_weight = load_pickle(os.path.join(experiment_directory, 'G_weights_{}_{}_{}.torch'.format(args.algos, dataset, args.weight_index)))
    algo.G.load_state_dict(G_weight)

    with torch.no_grad():
        steps = 32    
        x_past = x_real_block[:, :base_config.p, :].clone().to(base_config.device)
        
        x_fake_future = algo.G.sample(steps, x_past)

        if isinstance(x_fake_future, Tuple):
            x_fake = x_fake_future[0]
        else:
            x_fake = x_fake_future
        
        #print("fake size :", x_fake.shape)

        num_windows, window_size, num_vars = x_fake.shape
        T = num_windows + window_size - 1
        
        sums   = torch.zeros(T, num_vars, device=x_fake.device)
        counts = torch.zeros(T, 1,        device=x_fake.device)
        
        for w in range(num_windows):
            for s in range(window_size):
                g = w + s  # global time index
                sums[g]   += x_fake[w, s]
                counts[g] += 1
        
        time_series = sums / counts.clamp_min(1.0)   # (5000, 4)
        gan_ts = time_series.detach().cpu().numpy()

        np.save(os.path.join(experiment_directory, 'gen_{}_{}_{}.npy'.format(args.algos, dataset, args.index)), gan_ts)
        print('complete generation')
        

def run(algo_id, base_config, base_dir, dataset, spec, data_params={}):
    """ Create the experiment directory, calibrate algorithm, store relevant parameters. """
    print('Executing: %s, %s, %s' % (algo_id, dataset, spec))
    
    experiment_directory = pt.join(base_dir, dataset, spec, 'seed={}'.format(base_config.seed), algo_id)
    if not pt.exists(experiment_directory):
        # if the experiment directory does not exist we create the directory
        os.makedirs(experiment_directory)
    base_config.experiment_directory = experiment_directory

    # set_seed(base_config.seed)
    pipeline, x_real, x_real_block = get_data(dataset, base_config.p, base_config.q, **data_params)
    
    # set algorithm
    algo = get_algo(algo_id, base_config, dataset, data_params, x_real_block)
    
    # Train the algorithm
    algo.fit()
    
    # Pickle generator weights, real path and hqyperparameters.
    pickle_it(algo.training_loss, pt.join(experiment_directory, 'training_loss.pkl'))
    pickle_it(algo.G.to('cpu').state_dict(), pt.join(experiment_directory, 'G_weights_{}_{}_{}.torch'.format(args.algos, dataset, args.weight_index)))
    pickle_it(algo.D.to('cpu').state_dict(), pt.join(experiment_directory, 'D_weights_{}_{}_{}.torch'.format(args.algos, dataset, args.weight_index)))
    
    # Log some results at the end of training
    algo.plot_losses()
    plt.savefig(os.path.join(experiment_directory, 'losses_{}_{}_{}.png'.format(args.algos, dataset, args.weight_index)))
        

def get_dataset_configuration(dataset):
    if dataset in ['sine', 'square', 'triangle', 'sawtooth', 'etth1', 'etth2', 'ettm1', 'ettm2', 'lorenz', 'licor']:
        generator = [('a', dict())]
    elif dataset in ['ILI', 'us_births'] :
        generator = [('a', dict())]
    else:
        raise Exception('%s not a valid data type.' % dataset)
    return generator


def main(args):
    if not pt.exists('./data'):
        os.mkdir('./data')

    args.use_cuda = True
    print('Start of training. CUDA: %s' % args.use_cuda)
    for dataset in args.datasets:
        hidden_dims = args.hidden_dims
        for algo_id in args.algos:
            for seed in range(args.initial_seed, args.initial_seed + args.num_seeds):
                base_config = BaseConfig(
                    device='cuda:{}'.format(args.device) if args.use_cuda and torch.cuda.is_available() else 'cpu',
                    seed=seed,
                    batch_size=args.batch_size,
                    hidden_dims=hidden_dims,
                    p=args.p,
                    q=args.q,
                    total_steps=args.total_steps,
                    mc_samples=1000,
                    eps=args.eps,
                    noise_type=args.noise_type,
                    use_ec=args.use_ec,
                )
                # set_seed(seed)
                generator = get_dataset_configuration(dataset)
                for spec, data_params in generator:
                    if args.test:
                        run_test(
                            algo_id=algo_id,
                            base_config=base_config,
                            data_params=data_params,
                            dataset=dataset,
                            base_dir=args.base_dir,
                            spec=spec,
                        )
                    elif args.mcmc:
                        run_mcmc(
                            algo_id=algo_id,
                            base_config=base_config,
                            data_params=data_params,
                            dataset=dataset,
                            base_dir=args.base_dir,
                            spec=spec,
                            )
                    else:
                        run(
                            algo_id=algo_id,
                            base_config=base_config,
                            data_params=data_params,
                            dataset=dataset,
                            base_dir=args.base_dir,
                            spec=spec,
                        )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # Meta parameters
    parser.add_argument('-base_dir', default='./results', type=str)
    parser.add_argument('-use_cuda', action='store_true')
    parser.add_argument('-device', default=0, type=int)
    parser.add_argument('-num_seeds', default=1, type=int)
    parser.add_argument('-initial_seed', default=0, type=int)
    parser.add_argument('-datasets', default=['etth1'], nargs="+")
    parser.add_argument('-algos', default=['AECGAN'], nargs="+")


    # Algo hyperparameters
    parser.add_argument('-batch_size', default=16, type=int)
    parser.add_argument('-p', default=168, type=int)
    parser.add_argument('-q', default=168, type=int)
    parser.add_argument('-hidden_dims', default=3 * (50,), type=tuple)
    parser.add_argument('-total_steps', default=10000, type=int)
    parser.add_argument('-noise_type', default='min_adv', type=str)
    parser.add_argument('-use_ec', default=2, type=int)
    parser.add_argument('-weight_index', default=0, type=int)

    # other 
    parser.add_argument('-eps', default=0.01, type=float)
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-mcmc', action='store_true', help='Enable MCMC selection stage')
    parser.add_argument('-index', default=0, type=int)
    
    args = parser.parse_args()
    main(args)
