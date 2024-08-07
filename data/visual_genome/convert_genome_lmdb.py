import os
import sys

import csv
import pickle
import lmdb  # install lmdb by "pip install lmdb"
import argparse
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def convert_to_lmdb(infiles, save_path):
    env = lmdb.open(save_path, map_size=1099511627776, writemap=True)

    id_list = []
    with env.begin(write=True) as txn:
        for infile in infiles:
            with open(infile) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
                for item in tqdm(reader):
                    img_id = str(item['img_id']).encode()
                    # str(int(item['img_id']).split('_')[-1])).encode()
                    # print(img_id)
                    # exit(0)
                    id_list.append(img_id)
                    txn.put(img_id, pickle.dumps(item))
        # print(len(id_list))
        txn.put('keys'.encode(), pickle.dumps(id_list))


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert to LMDB')
    parser.add_argument('--indir', type=str, default='/root/autodl-tmp/datasets/BLA/')
    # data/mscoco/imgfeats')
    parser.add_argument('--outdir', type=str, default='/root/autodl-tmp/datasets/BLA/imgfeats/volta')
    parser.add_argument('--split', type=str, default='genome', choices=['coco-vist', 'coco', 'vist', 'genome'])

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Setup the configuration, normally do not need to touch these:
    args = parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if args.split == 'coco-vist':
        args.infiles = [os.path.join(args.indir, "coco-imgs-features.tsv"),
                        os.path.join(args.indir, "vist-imgs-features.tsv")]
    elif args.split == 'coco':
        args.infiles = [os.path.join(args.indir, "coco-imgs-features.tsv")]
    elif args.split == 'vist':
        args.infiles = [os.path.join(args.indir, "vist-imgs-features.tsv")]
    elif args.split == 'genome':
        args.infiles = [os.path.join(args.indir, "genome-imgs-features.tsv")]
    
    args.outpath = os.path.join(args.outdir, "%s_feat.lmdb" % args.split)
    print('Called with args:')
    print(args)

    # Convert to LMDB
    convert_to_lmdb(args.infiles, args.outpath)