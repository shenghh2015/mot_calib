import os

def main():

    root_dir = os.path.abspath('../')
    # dataset_name = 'dancetrack'
    dataset_name = 'MOT20'
    subset = 'train'
    # subset = 'test'

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
        print(f'process {seq_name}')
        calib_command = f'python gen_sequence_video.py --dataset_name {dataset_name} --seq_name {seq_name} --use_gt --subset {subset}'
        os.system(calib_command)
        
        if i == 6: break
        if i < 6:
            calib_command = f'python gen_sequence_video.py --dataset_name {dataset_name} --seq_name {seq_name} --use_gt --subset {subset} --use_full'
            os.system(calib_command)

if __name__ == '__main__':
    main()