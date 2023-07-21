# import nltk
from tqdm import tqdm
import pickle
import json
import jsonlines
import random
# nltk.download('punkt')

# original, ablation
dataset_type = "original" 

def read_sentences(jsonfiles, save_path):
    selected_samples = []
    for i, jsonfile in enumerate(jsonfiles):
        with jsonlines.open(jsonfile) as reader:
            n_samples = 34 if i == 2 else 33
            selected_samples.extend(random.sample(list(reader), n_samples))
    
    random.shuffle(selected_samples)
    
    with jsonlines.open(save_path, mode='w') as writer:
        for elem, s in tqdm(enumerate(selected_samples)):
            d = {'sentences': s['sentences'], 'id': s['id'], 'img_path': s['img_path']}   
            writer.write(d)
            

def main(mysentences, myjsonl):
    read_sentences(mysentences, myjsonl)


if __name__ == '__main__':
    # types = ["finetune_random", "finetune_test_golden", "finetune_train_golden"]
    types =["finetune_random"]
    tasks = ["active_passive", "coord", "rc"]
    mysentences = []
    for split_type in types:
        for task_type in tasks:
            mysentences.append('/home/xchen/datasets/BLA_finetune_0215/{}/{}/train_ann.jsonl'.format(split_type, task_type))
        myjsonl = '/home/xchen/datasets/BLA_finetune_0215/{}/mix_train_ann.jsonl'.format(split_type, tasks)
        main(mysentences, myjsonl)
    
    print("finish")