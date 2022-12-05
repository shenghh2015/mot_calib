import os

def main():

    root_dir = os.path.abspath('../')
    # dataset_name = 'MOT20'
    dataset_name = 'mot'
    subset = 'train'
    use_gt = True

    data_dir = os.path.join(root_dir, 'datasets', dataset_name, subset)

    if dataset_name == 'mot':
        seq_list = [seq for seq in os.listdir(data_dir) if 'FRCNN' in seq and 'MOT17' in seq]
    elif dataset_name == 'MOT20':
        seq_list = [seq for seq in os.listdir(data_dir) if 'MOT20' in seq]
    else:
        seq_list = [seq for seq in os.listdir(data_dir) if 'dancetrack' in seq]
    
    print(seq_list)

    # seq_name = seq_list[0]

    for seq_name in seq_list:
        print(f'calibrate {seq_name}')
        calib_command = f'python calib_process_v2.py --dataset_name {dataset_name} --seq_name {seq_name} --subset {subset}'
        if use_gt: calib_command += ' --use_gt'
        os.system(calib_command)

if __name__ == '__main__':
    main()