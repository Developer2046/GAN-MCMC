# GAN-MCMC

This repository contains the code of **Preserving Temporal Dynamics in Time Series Generation**.

Getting Started:
1. Prepare the data:
    - We have provide the data in the folder ``data``, containing the following seven datasets:
        - ``[etth1, etth2, ettm1, ettm2, lorenz, Licor, ILI]``
2. Install dependencies:
    - This project is implemented with ``pytorch==1.8.1+cu102``
3. For training
    ```bash
    python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 0
    ```
    - ``$dataset``: We have implemented ``AEC-GAN`` on seven datasets: ``[etth1, etth2, ettm1, ettm2, lorenz, Licor, ILI]``
    - ``$save_path``: The path you save the model.
    - ``$p``: The length of the past conditions.
    - ``$q``: The length of the forward generations.
4. For generation 
    ```bash
    python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 0 -test -index 0
    python train.py -datasets 'etth1'     -base_dir 'results/p16_q32' -p 16 -q 32 -use_cuda -algos 'AECGAN' -total_steps 20000 -batch_size 100 -noise_type min_adv -use_ec 2 -weight_index 0 -mcmc -index 0
    ```
    - ``$dataset``: We have implemented ``AEC-GAN`` on six datasets: ``[etth1, etth2, ettm1, ettm2, us_births, ILI]``
    - ``$save_path``: The path you save the model.
    - ``$p``: The length of the past conditions.
    - ``$q``: The length of the forward generations.
    - ``$mcmc``:The generated data will be corrected by MCMC process.
    - ``$test``:Generated data in a normal manner.
5. For downstream performance
    - The generated data can be used as the alternative training set used for training forecasting models (e.g., SCINet, Informer and Autoformer).
    - Folder `models` contains the code to train the downstream forecasting models.
6. Easy usage
    - For an easy usage, we also provide a bash file ``run_file.sh``, which contains the commands to train the models or generate time-series data.

Resources
- The GANs' code is partially based on the https://github.com/HBhswl/AEC-GAN.
