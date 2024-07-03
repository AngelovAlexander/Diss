import os, shutil
import argparse
import random

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='split_dir', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--train_dir', default="/", type=str)
    parser.add_argument('--valid_dir', default="/", type=str)
    parser.add_argument('--train_categories', default=644, type=int)
    parser.add_argument('--valid_categories', default=683, type=int)

    args = parser.parse_args()
    train_categories_dir = os.listdir(args.train_dir)
    valid_categories_dir = os.listdir(args.valid_dir)
    
    train_id_subset = random.choices(train_categories_dir, k=args.train_categories)
    valid_id_subset = random.choices(list(set(train_categories_dir) - set(train_id_subset)), k=args.valid_categories - args.train_categories)
    valid_id_subset.extend(train_id_subset)
    
    cur_dir_path = os.path.dirname(os.path.realpath(__file__))

    create_folder(cur_dir_path + "/Data")
    create_folder(cur_dir_path + "/Data/herbarium19/small-validation")
    create_folder(cur_dir_path + "/Data/herbarium19/small-train")
    
    for dir_id in valid_id_subset:
        shutil.copytree(args.valid_dir + dir_id, cur_dir_path + "/Data/herbarium19/small-validation/" + dir_id, dirs_exist_ok=True)
        if dir_id in train_id_subset:
            shutil.copytree(args.train_dir + dir_id, cur_dir_path + "/Data/herbarium19/small-train/" + dir_id, dirs_exist_ok=True)