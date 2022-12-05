import os
import argparse

def get_params():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type = str, default = 'mot')
    parser.add_argument('--subset', type = str, default = 'train')
    args   = parser.parse_args()
    return args


def main():

    root_dir = os.path.abspath('../')
    # dataset_name = 'dancetrack'
    # dataset_name = 'MOT20'
    # subset = 'train'
    # subset = 'test'

    args = get_params()

    dataset_name = args.dataset_name
    subset       = args.subset

    data_dir = os.path.join(root_dir, 'datasets', dataset_name, subset)

    if dataset_name == 'mot':
        seq_list = [seq for seq in os.listdir(data_dir) if 'FRCNN' in seq and 'MOT17' in seq]
    elif dataset_name == 'MOT20':
        seq_list = [seq for seq in os.listdir(data_dir) if 'MOT20' in seq]
    else:
        seq_list = [seq for seq in os.listdir(data_dir) if 'dancetrack' in seq]
    
    print(seq_list)

    # seq_name = seq_list[0]
    for i, seq_name in enumerate(seq_list):
        
        calib_command = f'python gen_sequence_video.py --dataset_name {dataset_name} --seq_name {seq_name} --subset {subset} --use_full'
        if subset == 'train' or subset == 'val':
            calib_command += ' --use_gt'
        os.system(calib_command)

if __name__ == '__main__':
    main()

## run example
## python run_video_gen2.py --dataset_name dancetrack --subset val