import random
import re

jsonl_root_path = '/root/autodl-tmp/datasets/BLA_finetune_0215'
jsonl_src_path = jsonl_root_path + '/{}/{}/{}_ann.jsonl'
jsonl_target_path = jsonl_root_path + '/{}/{}/{}_ann_{}_{}.jsonl'

yaml_root_path = '/root/autodl-tmp/volta/exmaple_xinyi_bla_train/task_configs/'

finetune_types = ['finetune_random', 'finetune_test_golden', 'finetune_train_golden']
dataset_types = ['active_passive', 'coord', 'rc']
task_types = ['test', 'train', 'valid']

default_split_sizes = [20, 50, 100]
default_times = 3


def read_lines(file_path):
    lines = []
    with open(file_path, "r") as file:
        while True:
            line = file.readline()
            if not line:
                break
            lines.append(line)
    return lines


def write_lines_to_file(file_path, lines):
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(line)


def split_jsonl(finetune_type, dataset_type, task_type, split_sizes, times):
    jsonl_lines = read_lines(jsonl_src_path.format(finetune_type, dataset_type, task_type))
    for size in split_sizes:
        if size > len(jsonl_lines):
            break
        for time in range(1, times + 1):
            select_lines = random.sample(jsonl_lines, size)
            write_lines_to_file(jsonl_target_path.format(finetune_type, dataset_type, task_type, size, time),
                                select_lines)


def change_task_yaml(yaml_file_path, task_type, sizes, times):
    for size in sizes:
        for time in range(1, times + 1):
            yaml_lines = read_lines(yaml_file_path)
            jsonl_file_name = task_type + '_ann.jsonl'
            for i in range(len(yaml_lines)):
                line = yaml_lines[i]
                if "annotations_jsonpath" in line and jsonl_file_name in line:
                    yaml_lines[i] = line.replace(jsonl_file_name, task_type + '_ann_{}_{}.jsonl'.format(size, time))
            write_lines_to_file(re.sub(".yml$", '_{}_{}.yml'.format(size, time), yaml_file_path), yaml_lines)


def main():
    split_jsonl('finetune_random', 'coord', 'train', default_split_sizes, 1)
    change_task_yaml(yaml_root_path + 'ctrl_coord_tasks.yml', 'train', default_split_sizes, 1)
    split_jsonl('finetune_random', 'rc', 'train', default_split_sizes, 1)
    change_task_yaml(yaml_root_path + 'ctrl_rc_tasks.yml', 'train', default_split_sizes, 1)


if __name__ == '__main__':
    main()
