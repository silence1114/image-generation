import argparse
import os
import tempfile
import tensorflow as tf
import numpy as np
import tfimage as im

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default='train_data/crop_data/128pad256/', help="path to folder containing images")
parser.add_argument("--output_dir", default='train_data/combine_data/128to256/', help="output path")
parser.add_argument("--b_dir", default='train_data/crop_data/256/', help="path to folder containing B images for combine operation")
a = parser.parse_args()


def combine(src, src_path):
    basename, _ = os.path.splitext(os.path.basename(src_path))
    for ext in [".png", ".jpg"]:
        sibling_path = os.path.join(a.b_dir, basename + ext)
        if os.path.exists(sibling_path):
            sibling = im.load(sibling_path)
            break

    return np.concatenate([src, sibling], axis=1)


def main():
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    src_paths = []
    dst_paths = []

    for src_path in im.find(a.input_dir):
        name, _ = os.path.splitext(os.path.basename(src_path))
        dst_path = os.path.join(a.output_dir, name + ".png")
        src_paths.append(src_path)
        dst_paths.append(dst_path)


    with tf.Session() as sess:
        for src_path, dst_path in zip(src_paths, dst_paths):
            src = im.load(src_path)
            dst = combine(src, src_path)
            im.save(dst, dst_path)

    
main()
