# import nltk
from tqdm import tqdm
import pickle
import json
import jsonlines
# nltk.download('punkt')

mysentences = '/root/autodl-tmp/datasets/FOIL/foil_debug.json'
myjsonl = '/root/autodl-tmp/datasets/FOIL/annotations_debug/{}_ann.jsonl'.format(mysentences.split('/')[-1][:-5])

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
    with jsonlines.open(save_path, mode='w') as writer:
        for elem, items in tqdm(list(enumerate(data.items()))):
            id, s = items 
            sentences = s["captions"]
            img_id = s["img_name"]
            d = {'sentences': sentences, 'id': str(img_id), 'img_path': img_id}   
            writer.write(d) 

def main():
    read_sentences(mysentences,myjsonl)


if __name__ == '__main__':
    main()