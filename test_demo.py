import argparse
from collections import defaultdict
from datetime import datetime

import time
import yaml
from tqdm import tqdm

from get_instances import *
from utils import *

import matplotlib.pyplot as plt

def initArgs(arguments):
    config_path = arguments.config
    with open(config_path, "r") as fr:
        configs = yaml.load(fr, Loader=yaml.FullLoader)
    device = arguments.device

    # Read configs
    n_layers = configs['n_layers']
    k_iters = configs['k_iters']

    n = configs['n']

    dataset_name = configs['dataset_name']
    dataset_params = configs['dataset_params']
    dataset_params['samplerate'] = arguments.samplerate

    batch_size = configs['batch_size'] if arguments.batch_size is None else arguments.batch_size

    model_name = configs['model_name']
    model_params = configs.get('model_params', {})
    model_params['n_layers'] = n_layers
    model_params['k_iters'] = k_iters
    model_params['phconv_params'] = n

    score_names = configs['score_names']

    config_name = configs['config_name']

    workspace = os.path.join(arguments.workspace, config_name)  # workspace/config_name
    checkpoints_dir, log_dir = get_dirs(workspace)  # workspace/config_name/checkpoints ; workspace/config_name/log.txt

    dataloader = get_loaders(dataset_name, dataset_params, batch_size, ['test'])['test']
    model = get_model(model_name, model_params, device)
    score_fs = get_score_fs(score_names)

    # restore
    saver = CheckpointSaver(checkpoints_dir)
    prefix = 'best' if configs['val_data'] else 'final'
    checkpoint_path = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.startswith(prefix)][0]
    model = saver.load_model(checkpoint_path, model, device)

    return configs, device, workspace, dataloader, model, score_fs

def main(arguments):
    configs, device, workspace, dataloader, model, score_fs = initArgs(arguments)
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('test start: ' + start_time)

    new_folder_path = f"./outputImages/{start_time}' '{configs['description']}"
    os.makedirs(new_folder_path, exist_ok=True)

    start = time.time()

    running_score = defaultdict(int)

    model.eval()
    for i, (x, y, csm, mask) in enumerate(tqdm(dataloader)):
        x, csm, mask = x.to(device), csm.to(device), mask.to(device)

        with torch.no_grad():
            y_pred = model(x, csm, mask).detach().cpu()

        y = np.abs(r2c(y.numpy(), axis=1))
        y_pred = np.abs(r2c(y_pred.numpy(), axis=1))
        for score_name, score_f in score_fs.items():
            running_score[score_name] += score_f(y, y_pred) * y_pred.shape[0]
        if arguments.write_image > 0 and (i % arguments.write_image == 0):
            display_img(
                np.abs(r2c(x[-1].detach().cpu().numpy())),
                mask[-1].detach().cpu().numpy(),
                y[-1],
                y_pred[-1],
                psnr(
                    y[-1],
                    y_pred[-1]
                )
            )

            plt.savefig(os.path.join(new_folder_path, f'output{i}.png'))
            plt.close()

    epoch_score = {score_name: score / len(dataloader.dataset) for score_name, score in running_score.items()}
    for score_name, score in epoch_score.items():
        print('test {} score: {:.4f}'.format(score_name, score))

    print('-----------------------')
    print('total test time: {:.4f} sec'.format((time.time()-start)/3600))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, required=False, default="configs/test_hyper, k=1.yaml",
                        help="config file path")
    parser.add_argument("--device", type=str, required=False, default="cpu",
                        help="[cpu / cuda]")
    parser.add_argument("--workspace", type=str, default='./workspace')
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--write_image", type=int, default=1)
    parser.add_argument("--samplerate", type=int, default=8)

    args = parser.parse_args()

    main(args)