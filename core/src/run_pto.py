import os
import shutil
import random
from argparse import Namespace, ArgumentParser

import time
import yaml
import numpy as np
import torch

from allocate.experiment_pto import PtOExperiment

def run_pto_experiment(configs):
    # Fix random seed to ensure reproducibility
    random.seed(configs.random_seed)
    np.random.seed(configs.random_seed)
    torch.manual_seed(configs.random_seed)

    # Instantiate experiment manager
    experiment = PtOExperiment(configs)
    
    # Evaluate
    print(f'{">" * 20} {"Start testing:":<15} {configs.exp_id} {"<" * 20}')
    start = time.time()
    metrics = experiment.evaluate(experiment.test_loader, experiment.test_set.scaler, save_result=True)
    print(f'Time: {(time.time() - start) / 2:.2f}')
    print(f'Test MSE: {metrics["MSE"]:.4f} | Test MAE: {metrics["MAE"]:.4f} | Test Regret: {metrics["Regret"]:.8f} | Test Rel Regret: {metrics["Rel Regret"]:.8f}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as fin:
        configs = yaml.safe_load(fin)

    # Remove the previous experiment with the same `exp_id`.
    exp_dir = os.path.join('output', configs['Experiment']['exp_id'])
    if os.path.exists(exp_dir):
        print(f'Experiment {exp_dir} exists, delete and continue? [Y/N]', end=' ')

        response = input()
        while response not in ['Y', 'N']:
            print('Invalid choice. Choose between [Y/N]', end=' ')
            response = input()

        shutil.rmtree(exp_dir) if response == 'Y' else exit()

    os.makedirs(exp_dir)

    # Copy config file to output directory.
    config_path = os.path.join(exp_dir, 'config.yaml')
    with open(config_path, 'w') as fout:
        yaml.dump(configs, fout, indent=4, sort_keys=False)

    configs = Namespace(**{
        arg: val
        for _, args in configs.items()
        for arg, val in args.items()
    })
    
    run_pto_experiment(configs)
