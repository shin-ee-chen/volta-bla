import os
import sys

import csv
import pickle
import lmdb  # install lmdb by "pip install lmdb"
import argparse
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

# FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
#               "attrs_id", "attrs_conf", "num_boxes", "boxes", "features",
#               "cls_prob", "attrs", "classes"]
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def convert_to_lmdb(infiles, save_path):
    save_dir = '/'.join(save_path.split('/')[:-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    env = lmdb.open(save_path, map_size=1099511627776)
    id_list = []
    with env.begin(write=True) as txn:
        for infile in infiles:
            with open(infile) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
                for item in tqdm(reader):
                    img_id = str(item['img_id']).encode()
                    id_list.append(img_id)
                    txn.put(img_id, pickle.dumps(item))
        txn.put('keys'.encode(), pickle.dumps(id_list))


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert to LMDB')
    parser.add_argument("--indir", type=str, 
                        # default= "/home/xinyi/volta/data/flickr30k/imgfeats",
                        default="/home/xinyi/datasets/BLA_finetune/",
                        help="Diretory of input file")
    parser.add_argument("--outdir", type=str, 
                        # default= "/home/xinyi/volta/data/flickr30k/imgfeats/volta",
                        default="/home/xinyi/datasets/BLA_finetune/imgfeats/volta/",
                        help="Diretory of input file")
    parser.add_argument('--split', type=str, default= "test", choices=['flickr30k', 'trainval', 'test'])

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Setup the configuration, normally do not need to touch these:
    args = parse_args()
    
    if args.split == 'flickr30k':
        args.infiles = [os.path.join(args.indir, "flickr30k_obj36-36.tsv")]
    elif args.split == 'trainval':
        args.infiles = [os.path.join(args.indir, "train_obj36-36.tsv"),
                        os.path.join(args.indir, "valid_obj36-36.tsv")]
    else:
        # args.infiles = [os.path.join(args.indir, "test_obj36-36.tsv")]
        args.infiles = [os.path.join(args.indir, "genome-imgs-features.tsv")]
    args.outpath = os.path.join(args.outdir, "%s_feat.lmdb" % args.split)
    
    print('Called with args:')
    print(args)

    # Convert to LMDB
    convert_to_lmdb(args.infiles, args.outpath)

