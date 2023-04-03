# import nltk
from tqdm import tqdm
import pickle
import json
import jsonlines
# nltk.download('punkt')
import os

mysentences = '/root/autodl-tmp/datasets/BLA_finetune/annotations/finetune_random/active_passive/test.json'
myjsonl = '/root/autodl-tmp/datasets/BLA_finetune_0215/finetune_random/active_passive/{}_ann.jsonl'.format(
    mysentences.split('/')[-1][:-5])


def read_json_file(file_path):
    global json_file
    try:
        json_file = open(file_path, "r")
        return json.load(json_file)
    finally:
        if json_file:
            print("close file...")
            json_file.close()


def read_sentences(jsonfile, save_path):
    data = read_json_file(jsonfile)
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with jsonlines.open(save_path, mode='w') as writer:
        for elem, items in tqdm(list(enumerate(data.items()))):
            id, s = items
            sentences = s
            img_id = id
            d = {'sentences': sentences, 'id': str(img_id), 'img_path': img_id}
            writer.write(d)


src_path = '/root/autodl-tmp/datasets/BLA_finetune/annotations/{}/{}/{}.json'
target_path = '/root/autodl-tmp/datasets/BLA_finetune_0215/{}/{}/{}_ann.jsonl'

task_types = ['finetune_random', 'finetune_test_golden', 'finetune_train_golden']
dataset_types = ['active_passive', 'coord', 'rc']
file_types = ['test', 'train', 'valid']


def main():
    for task_type in task_types:
        for dataset_type in dataset_types:
            for file_type in file_types:
                input_file = src_path.format(task_type, dataset_type, file_type)
                output_file = target_path.format(task_type, dataset_type, file_type)
                read_sentences(input_file, output_file)


if __name__ == '__main__':
    main()
