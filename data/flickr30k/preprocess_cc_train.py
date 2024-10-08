import os
import time
import numpy as np
from tensorpack.dataflow import *
import json
import csv
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
            # ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
            #   "attrs_id", "attrs_conf", "num_boxes", "boxes", "features",]
            #   "cls_prob", "attrs", "classes"]
import sys
import pandas as pd
import zlib
import base64

csv.field_size_limit(sys.maxsize)

split = "test"
task = "rc"
type = "finetune_train_golden_large"

# corpus_path = "/home/xinyi/datasets/BLA_finetune_bak"
# infile_tsv_path =  os.path.join(corpus_path, "genome-imgs-features.tsv")
# infile_json_path = os.path.join(corpus_path, "annotations", type, task, split + '.json')
# outfile_lmdb_path = os.path.join(corpus_path, 'imgfeats/volta/', type, task, split + '.lmdb')

def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', names=["caption", "url"], usecols=range(0, 2))
    df['folder'] = folder
    print("Processing", len(df), " Images:")
    return df


def _file_name(row):
    return "%s/%s" % (row['folder'], (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))


class Conceptual_Caption(RNGDataFlow):
    """
    """
    def __init__(self, infile_tsv_path, infile_json_path, shuffle=False):
        """
        Same as in :class:`ILSVRC12`.
        """
        self.shuffle = shuffle
        # self.name = os.path.join(corpus_path, 'genome-imgs-features.tsv')
        self.name =  infile_tsv_path
        self.infiles = [self.name]
        self.counts = []

        self.captions = {}
        # df = open_tsv(os.path.join(corpus_path, 'genome-imgs-features.tsv'), 'training')
        # for i, img in enumerate(df.iterrows()):
        #     caption = img[1]['caption']
        #     url = img[1]['url']
        #     im_name = _file_name(img[1])
        #     image_id = im_name.split('/')[1]
        #     self.captions[image_id] = caption

        with open(infile_json_path) as f:
            captions = json.load(f)
            num_sentences = 1
            
        for img_id, sentences in captions.items():
            self.captions[img_id] = sentences
            num_sentences = len(sentences)
            
        self.num_caps = len(captions) * num_sentences

    def __len__(self):
        return self.num_caps

    def __iter__(self):
        for infile in self.infiles:
            count = 0
            # remove existing infile to avid errors
            
            with open(infile) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
                for item in reader:
                    time.sleep(0.005)
                    image_id = item['img_id']
                    if image_id not in self.captions:
                        continue              
                    image_h = item['img_h']
                    image_w = item['img_w']
                    num_boxes = item['num_boxes']
                    boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(int(num_boxes), 4)
                    features = np.frombuffer(base64.b64decode(item['features']), dtype=np.float32).reshape(int(num_boxes), 2048)
                    # cls_prob = np.frombuffer(base64.b64decode(item['cls_prob']), dtype=np.float32).reshape(int(num_boxes), 1601)
                    objects_id = np.frombuffer(base64.b64decode(item['objects_id']), dtype=np.int64)
                    objects_conf = np.frombuffer(base64.b64decode(item['objects_conf']), dtype=np.float32)
                    attrs_id = np.frombuffer(base64.b64decode(item['attrs_id']), dtype=np.int64)
                    attrs_conf = np.frombuffer(base64.b64decode(item['attrs_conf']), dtype=np.float32)
                    # cls_scores = np.frombuffer(base64.b64decode(item['classes']), dtype=np.float32).reshape(int(num_boxes), -1)
                    # attr_scores = np.frombuffer(base64.b64decode(item['attrs']), dtype=np.float32).reshape(int(num_boxes), -1)
                    for i, sentence in enumerate(self.captions[image_id]):
                        caption = sentence
                        label = 1 if i < 1 else 0
                        # yield [features, cls_prob, objects_id, objects_conf, attrs_id, attrs_conf, attr_scores, boxes, num_boxes, image_h, image_w, image_id, caption]
                        yield [features, objects_id, objects_conf, attrs_id, attrs_conf, boxes, num_boxes, image_h, image_w, image_id, caption, label]


if __name__ == '__main__':
    # types = ["finetune_random", "finetune_test_golden", "finetune_train_golden"]
    tasks = ["active_passive", "coord", "rc"]
    splits = ["train", "valid", "test"]
    
    corpus_path = "/root/autodl-tmp/datasets/BLA_finetune_bak"
    infile_tsv_path = "/root/autodl-tmp/datasets/BLA/genome-imgs-features.tsv"
    for type in types:
        for task in tasks:
            for split in splits:
                infile_json_path = os.path.join(corpus_path, "annotations", type, task, split + '.json')
                outfile_lmdb_path = os.path.join(corpus_path, 'imgfeats/volta/', type, task, split + '.lmdb')

                ds = Conceptual_Caption(infile_tsv_path, infile_json_path)
                ds1 = PrefetchDataZMQ(ds, nr_proc=1)
                if os.path.isfile(outfile_lmdb_path):
                    os.remove(outfile_lmdb_path)
                if not os.path.exists(os.path.dirname(outfile_lmdb_path)):
                    os.makedirs(os.path.dirname(outfile_lmdb_path))
                LMDBSerializer.save(ds1, outfile_lmdb_path)
    
                print("Output to", outfile_lmdb_path, "num_entites:", len(ds1))
    print("Finish")
