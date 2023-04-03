# import nltk
from tqdm import tqdm
import pickle
import json
import jsonlines
# nltk.download('punkt')

# original, ablation
dataset_type = "original" 

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
    print(len(data))
    with jsonlines.open(save_path, mode='w') as writer:
        
        for elem, s in tqdm(list(enumerate(data))):
            if dataset_type == "original":
                if 'caption_group' in s:
                    captions = s['caption_group'][0]
                    sentences = [captions['True1'], captions['True2'],
                                captions['False1'], captions['False2']]
            else:
                    sentences = [s['sub_obj'], s['obj_sub']]

            img_id = s['image_id']
            d = {'sentences': sentences, 'id': str(img_id), 'img_path': str(img_id)+".jpg"}   
            writer.write(d) 

def main(mysentences, myjsonl):
    read_sentences(mysentences,myjsonl)


if __name__ == '__main__':
    types = ["finetune_random", "finetune_test_golden", "finetune_train_golden"]
    tasks = ["active_passive", "coord", "rc"]
    for split_type in types:
        for task_type in tasks:
            mysentences = '/home/xinyi/datasets/BLA_finetune/annotations/{}/{}.json'.format(split_type, task_type)
            myjsonl = '/home/xinyi/datasets/BLA_finetune/annotations/{}/{}_ann.jsonl'.format(split_type, task_type)
            main(mysentences, myjsonl)
    
    print("finish")