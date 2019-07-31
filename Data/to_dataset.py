import os
import shutil 
import random
import csv
import tqdm
import PIL
import pandas as pd
import numpy as np
import argparse
from autoaugment import ImageNetPolicy
from categorization import categorization

WORD2VEC = "./glove.6B.50d.txt"
FILE = "./Metadata.csv"

DATA_PATH_TRAIN = 'dataset/train/'
DATA_PATH_VALID = 'dataset/valid/'
DATA_PATH_TEST = 'dataset/test/'
TAG_PATH = 'npy-file/'

parser = argparse.ArgumentParser(description='Generating Dataset for Image and Tags')
parser.add_argument('--augment', action='store_true',
                    help='Image Auto Augmentation (default: false)')
parser.add_argument('--tag', type=str, default='blue',
                    help='tag to generate dataset (default: blue)')
parser.add_argument('--scale', type=int, default=2,
                    help='Augmentation scale (default: 2)')

# split images to train/valid/test in ./dataset
def data_split():
    # read table.csv
    os.chdir('../Data/')
    with open('table.csv', mode='r') as infile:
        reader = csv.reader(infile)
        mydict = {rows[0]:rows[1] for rows in reader}
        
    if args.tag not in categorization().keys() and args.tag != 'plat':
        print("Invalid tag")
        return

    try:
        shutil.rmtree('dataset')
    except:
        pass
    os.mkdir('dataset')

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

    os.mkdir('./dataset/train')
    os.mkdir('./dataset/train/y')
    os.mkdir('./dataset/train/n')
    for p in train:
        if p in imgs_list['y']:
            src = from_path+p
            dst = './dataset/train/y/'+p
            shutil.copyfile(src,dst)
        else:
            src = from_path+p
            dst = './dataset/train/n/'+p
            shutil.copyfile(src,dst)
    
    os.mkdir('./dataset/valid')
    os.mkdir('./dataset/valid/y')
    os.mkdir('./dataset/valid/n')
    for p in valid:
        if p in imgs_list['y']:
            src = from_path+p
            dst = './dataset/valid/y/'+p
            shutil.copyfile(src,dst)
        else:
            src = from_path+p
            dst = './dataset/valid/n/'+p
            shutil.copyfile(src,dst)

    os.mkdir('./dataset/test')
    os.mkdir('./dataset/test/y')
    os.mkdir('./dataset/test/n')
    for p in test:
        if p in imgs_list['y']:
            src = from_path+p
            dst = './dataset/test/y/'+p
            shutil.copyfile(src,dst)
        else:
            src = from_path+p
            dst = './dataset/test/n/'+p
            shutil.copyfile(src,dst)

    return imgs_list,train,valid,test

# autoaugmentation
def data_augmentation(scale=2):
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

# preprocess Metadata.csv
def preprocess(f):
    print('-'*10)
    print("Preprocessing Image_tag.....")
    print('-'*10)
    # preprocess categorization
    category = categorization()

    # preprocess Metadata.csv
    df = pd.read_csv(f, encoding = "ISO-8859-15", header=None, low_memory=False)
    df = df[[0,5]]
    # print(df.loc[df[0]== 4000000])
    
    img_tags = {}
    for _, row in tqdm.tqdm(df.iterrows()):
        img, string = row[0], row[5]
        try:
            string = string.split('   ')
            # remove UI tags
            string = [x for x in string if x not in category['ui_tags']]
            string = [x for x in string if x not in category[args.tag]]
            string = " ".join(string)
            string = string.split(' ')
            string = [x for x in string if len(x) > 1]
            string = [x for x in string if not x.isdigit()]
            string = list(set(string))
        except:
            string = []
        img_tags[img] = string
    # max 22 tags
    return img_tags

# load Word2Vec model
def loadGloveModel(gloveFile):
    print('-'*10)
    print("Loading Glove Model.....")
    print('-'*10)
    f = open(gloveFile,'r')
    model = {}
    for line in tqdm.tqdm(f):
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    # print("Done.",len(model)," words loaded!")
    return model

# # Generating Dataset for Tags CNN
# def generate_tag_cnn():
#     try:
#         img_list,train,valid,_ = data_split()
#     except:
#         return
#     if args.augment:
#         data_augmentation(args.scale)
#     img_tags = preprocess(FILE)
#     model = loadGloveModel(WORD2VEC)

#     npys_train = []
#     npys_valid = []
#     npys_test = []
#     for i in tqdm.tqdm(img_list['all']):
#         id = i.split('.')[0].split('_')[-1]
#         try:
#             tags = img_tags[int(id)]
#         except:
#             tags = []
#         vecs = []
#         for t in tags:
#             try:
#                 vec = model[t]
#                 vecs.append(vec)
#             except:
#                 pass
#         # completing vector to 1*500
#         for _ in range(50-len(vecs)):
#             vecs.append(np.zeros((50,), dtype=float))
#         vecs = np.asarray(vecs)
#         vecs = vecs.flatten()
#         npy = np.asarray([vecs,1 if i in img_list['y'] else 0])
        
#         if i in train:
#             npys_train.append(npy)
#         elif i in valid:
#             npys_valid.append(npy)
#         else:
#             npys_test.append(npy)
    
#     x_train = np.asarray(npys_train)
#     x_valid = np.asarray(npys_valid)
#     x_test = np.asarray(npys_test)
#     print(x_train.shape)

#     try:
#         shutil.rmtree(TAG_PATH)
#         os.mkdir(TAG_PATH)
#     except:
#         os.mkdir(TAG_PATH)
    
#     np.save(TAG_PATH+'x_train.npy', np.array(x_train))
#     np.save(TAG_PATH+'x_valid.npy', np.array(x_valid))
#     np.save(TAG_PATH+'x_test.npy', np.array(x_test))

if __name__ == "__main__":
    args = parser.parse_args()
    print('-'*10)
    print(args)
    print('-'*10)

    # Generating Dataset for Image and Tags
    data_split()
    if args.augment:
        data_augmentation(args.scale)