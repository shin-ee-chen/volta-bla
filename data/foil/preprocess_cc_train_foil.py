import os
import time
import numpy as np
from tensorpack.dataflow import *
import json
import csv
from collections import defaultdict

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



corpus_path = "/root/autodl-tmp/datasets/FOIL"
infile_tsv_path =  os.path.join(corpus_path, "foil-imgs-features.tsv")
infile_json_path = os.path.join(corpus_path, 'foil_debug.json')
outfile_lmdb_path = os.path.join(corpus_path, 'imgfeats_debug/volta/valid.lmdb')

# corpus_path = "/root/autodl-tmp/datasets/BLA_finetune"
# infile_tsv_path =  os.path.join(corpus_path, "genome-imgs-features.tsv")
# infile_json_path = "/root/autodl-tmp/datasets/BLA_finetune/annotations/toy/train.json"
# outfile_lmdb_path ="/root/autodl-tmp/datasets/BLA_finetune/imgfeats/volta/toy/train.lmdb"

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
    def __init__(self, corpus_path, shuffle=False):
        """
        Same as in :class:`ILSVRC12`.
        """
        self.shuffle = shuffle
        # self.name = os.path.join(corpus_path, 'genome-imgs-features.tsv')
        self.name =  infile_tsv_path
        self.infiles = [self.name]
        self.counts = []

        self.captions = defaultdict(lambda: defaultdict(list))
        self.num_caps = 0
        # df = open_tsv(os.path.join(corpus_path, 'genome-imgs-features.tsv'), 'training')
        # for i, img in enumerate(df.iterrows()):
        #     caption = img[1]['caption']
        #     url = img[1]['url']
        #     im_name = _file_name(img[1])
        #     image_id = im_name.split('/')[1]
        #     self.captions[image_id] = caption

        with open(infile_json_path) as f:
            captions = json.load(f)
            # num_sentences = 0
            
        for caption_idx, sentences in captions.items():
            img_id = sentences["img_name"][:-4]
            self.captions[img_id][caption_idx] = sentences["captions"]
            self.num_caps += len(sentences["captions"])
            

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
                    for caption_id, caption_grp in self.captions[image_id].items():
                        for i, sentence in enumerate(caption_grp):
                            correct_nr = len(caption_grp) / 2
                            caption = sentence
                            label = 1 if i < correct_nr else 0
                            # yield [features, cls_prob, objects_id, objects_conf, attrs_id, attrs_conf, attr_scores, boxes, num_boxes, image_h, image_w, image_id, caption]
                            image_location_wp = boxes
                            if image_location_wp.shape[0] != 36:
                                print(image_location_wp.shape, image_id)
                            yield [features, objects_id, objects_conf, attrs_id, attrs_conf, boxes, num_boxes, image_h, image_w, caption_id, caption, label]


if __name__ == '__main__':
    ds = Conceptual_Caption(corpus_path)
    ds1 = PrefetchDataZMQ(ds, nr_proc=1)
    # LMDBSerializer.save(ds1, os.path.join(corpus_path, 'imgfeats/volta/training_feat_all.lmdb'))
    
    if os.path.isfile(outfile_lmdb_path):
         os.remove(outfile_lmdb_path)
    if not os.path.exists(os.path.dirname(outfile_lmdb_path)):
        os.makedirs(os.path.dirname(outfile_lmdb_path))
    LMDBSerializer.save(ds1, outfile_lmdb_path)
    print("finish", "num_entites:", len(ds1))
