import os
import shutil 
import random
import csv
import tqdm
import PIL, cv2
import numpy as np
import argparse
from autoaugment import ImageNetPolicy
from categorization import categorization

parser = argparse.ArgumentParser(description='Generating Dataset for Image and Tags')
parser.add_argument('--auto', action='store_true',
                    help='Image Auto Augmentation (default: false)')
parser.add_argument('--noise', action='store_true',
                    help='Image Noise Augmentation (default: false)')
parser.add_argument('--tag', type=str, default='blue',
                    help='tag to generate dataset (default: blue)')
parser.add_argument('--scale', type=int, default=2,
                    help='Augmentation scale (default: 2)')
args = parser.parse_args()

# split images to train/valid/test in ./dataset
def data_split():
    global DATA_PATH_TRAIN, DATA_PATH_VALID, DATA_PATH_TEST
    DATA_PATH_TRAIN = args.tag+'/train/'
    DATA_PATH_VALID = args.tag+'/valid/'
    DATA_PATH_TEST = args.tag+'/test/'

    # read table.csv
    os.chdir('../Data/')
    with open('table.csv', mode='r') as infile:
        reader = csv.reader(infile)
        mydict = {rows[0]:rows[1] for rows in reader}
        
    if args.tag not in categorization().keys() and args.tag != 'plat':
        print("Invalid tag")
        return

    try:
        shutil.rmtree(args.tag)
    except:
        pass
    os.mkdir(args.tag)

    imgs_list = {}
    if args.tag == 'plat':
        imgs = os.listdir('./platform')
        imgs_list['y'] = [i for i in imgs if i.startswith('0_')]
        imgs_list['n'] = [i for i in imgs if i.startswith('1_')]
    else:
        imgs_list['y'] = [k for k,v in mydict.items() if args.tag in v]
        imgs_list['n'] = random.choices([k for k,v in mydict.items() if args.tag not in v], k=len(imgs_list['y']))
    
    imgs_list['all'] = imgs_list['y'] + imgs_list['n']
    random.shuffle(imgs_list['all'])
    random.shuffle(imgs_list['all'])

    # random split dataset
    ratio_train = 0.8
    ratio_validation = 0.1
    # ratio_testing = 0.1
    p1 = int(ratio_train * len(imgs_list['all']))
    p2 = int(ratio_validation * len(imgs_list['all']))
    train = imgs_list['all'][:p1]
    valid = imgs_list['all'][p1:p1+p2]
    test = imgs_list['all'][p1+p2:]

    if args.tag == 'plat':
        from_path = './platform/'
    else:
        from_path = './images/'

    paths = {DATA_PATH_TRAIN:train, DATA_PATH_VALID:valid, DATA_PATH_TEST:test}
    for k,v in paths.items():
        os.mkdir(k)
        os.mkdir(k+'y')
        os.mkdir(k+'n')
        for p in v:
            if p in imgs_list['y']:
                src = from_path+p
                dst = k+'y/'+p
                shutil.copyfile(src,dst)
            else:
                src = from_path+p
                dst = k+'n/'+p
                shutil.copyfile(src,dst)

    return imgs_list,train,valid,test

# autoaugmentation
def data_auto_augmentation(scale=2):
    print("Start Image augmentation...")
    path_y = DATA_PATH_TRAIN+'y/'
    save_dir_y = DATA_PATH_TRAIN+'y/'
    path_n = DATA_PATH_TRAIN+'n/'
    save_dir_n = DATA_PATH_TRAIN+'n/'

    j = 0
    for i in tqdm.tqdm(os.listdir(path_y)):
        try:
            for _ in range(scale):
                img = PIL.Image.open(path_y + i)#read
                policy = ImageNetPolicy()
                img1 = policy(img) #augmentation
                img1.save(save_dir_y + '{}.jpg'.format(j)) #creation
                j += 1
        except:
            pass 

    for i in tqdm.tqdm(os.listdir(path_n)):
        try:
            for _ in range(scale):
                img = PIL.Image.open(path_n + i)
                policy = ImageNetPolicy()
                img1 = policy(img)
                img1.save(save_dir_n + '{}.jpg'.format(j)) 
                j += 1
        except:
            pass 
    print("Done.....")

def add_gasuss_noise(image, mean=0, var=0.0001):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

def data_noise_augmentation():
    print("Start Image augmentation...")
    path_y = DATA_PATH_TRAIN+'y/'
    save_dir_y = DATA_PATH_TRAIN+'y/'
    path_n = DATA_PATH_TRAIN+'n/'
    save_dir_n = DATA_PATH_TRAIN+'n/'

    j = 0
    for i in tqdm.tqdm(os.listdir(path_y)):
        try:
            img = cv2.imread(path_y + i)
            img_out = add_gasuss_noise(img)
            cv2.imwrite(save_dir_y + '{}.jpg'.format(j), img_out)
            j += 1
        except:
            pass

    for i in tqdm.tqdm(os.listdir(path_n)):
        try:
            img = cv2.imread(path_n + i)
            img_out = add_gasuss_noise(img)
            cv2.imwrite(save_dir_n + '{}.jpg'.format(j), img_out)
            j += 1
        except:
            pass 
    print("Done.....")

if __name__ == "__main__":
    print('-'*10)
    print(args)
    print('-'*10)

    # Generating Dataset
    data_split()
    if args.noise:
        data_noise_augmentation()
    if args.auto:
        data_auto_augmentation(args.scale)