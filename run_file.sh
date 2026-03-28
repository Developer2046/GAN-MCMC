#!/bash/bin

# train the model one 
python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 0 
python train.py -datasets 'ILI'       -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 0
python train.py -datasets 'lorenz'    -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 0
python train.py -datasets 'licor'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 0

python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 1 
python train.py -datasets 'ILI'       -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 1
python train.py -datasets 'lorenz'    -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 1
python train.py -datasets 'licor'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 1

python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 2 
python train.py -datasets 'ILI'       -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 2
python train.py -datasets 'lorenz'    -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 2
python train.py -datasets 'licor'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 2

python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 3 
python train.py -datasets 'ILI'       -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 3
python train.py -datasets 'lorenz'    -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 3
python train.py -datasets 'licor'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 3

python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 4 
python train.py -datasets 'ILI'       -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 4
python train.py -datasets 'lorenz'    -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 4
python train.py -datasets 'licor'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 4

# generate data
python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 0 -test -index 0 
python train.py -datasets 'ILI'       -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 0 -test -index 0 
python train.py -datasets 'lorenz'    -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 0 -test -index 0 
python train.py -datasets 'licor'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 0 -test -index 0 

python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 0 -mcmc -index 0
python train.py -datasets 'ILI'       -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 0 -mcmc -index 0 
python train.py -datasets 'lorenz'    -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 0 -mcmc -index 0 
python train.py -datasets 'licor'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 0 -mcmc -index 0 

python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 1 -test -index 1 
python train.py -datasets 'ILI'       -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 1 -test -index 1 
python train.py -datasets 'lorenz'    -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 1 -test -index 1 
python train.py -datasets 'licor'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 1 -test -index 1 

python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 1 -mcmc -index 1 
python train.py -datasets 'ILI'       -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 1 -mcmc -index 1 
python train.py -datasets 'lorenz'    -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 1 -mcmc -index 1 
python train.py -datasets 'licor'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 1 -mcmc -index 1 

python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 2 -test -index 2 
python train.py -datasets 'ILI'       -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 2 -test -index 2 
python train.py -datasets 'lorenz'    -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 2 -test -index 2 
python train.py -datasets 'licor'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 2 -test -index 2 

python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 2 -mcmc -index 2 
python train.py -datasets 'ILI'       -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 2 -mcmc -index 2 
python train.py -datasets 'lorenz'    -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 2 -mcmc -index 2
python train.py -datasets 'licor'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 2 -mcmc -index 2 

python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 3 -test -index 3 
python train.py -datasets 'ILI'       -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 3 -test -index 3 
python train.py -datasets 'lorenz'    -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 3 -test -index 3 
python train.py -datasets 'licor'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 3 -test -index 3 

python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 3 -mcmc -index 3 
python train.py -datasets 'ILI'       -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 3 -mcmc -index 3 
python train.py -datasets 'lorenz'    -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 3 -mcmc -index 3
python train.py -datasets 'licor'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 3 -mcmc -index 3 

python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 4 -test -index 4 
python train.py -datasets 'ILI'       -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 4 -test -index 4 
python train.py -datasets 'lorenz'    -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 4 -test -index 4 
python train.py -datasets 'licor'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 4 -test -index 4 

python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 4 -mcmc -index 4 
python train.py -datasets 'ILI'       -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 4 -mcmc -index 4 
python train.py -datasets 'lorenz'    -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 4 -mcmc -index 4
python train.py -datasets 'licor'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 4 -mcmc -index 4 

