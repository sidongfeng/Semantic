import os
import random
import tqdm
import cv2
from PIL import Image
import numpy as np
import argparse
from autoaugment import ImageNetPolicy
from categorization import categorization,type__

FROM = '/Volumes/Macintosh HD/Users/charlie/Documents/All_images/'

parser = argparse.ArgumentParser(description='Generating Dataset for Image and Tags')
parser.add_argument('--auto', action='store_true',
                    help='Image Auto Augmentation (default: false)')
parser.add_argument('--noise', action='store_true',
                    help='Image Noise Augmentation (default: false)')
parser.add_argument('--tag', type=str, default='blue',
                    help='tag to generate dataset (default: blue)')
parser.add_argument('--scale', type=int, default=2,
                    help='Augmentation scale (default: 2)')
parser.add_argument('--resize', action='store_false',
                    help='Image resize (default: true)')
parser.add_argument('--resize_rate', type=int, default=224,
                    help='Image resize to 224*224 (default: 224)')
args = parser.parse_args()

# split images to train/valid/test in ./dataset
def data_split():
    if args.tag not in categorization().keys():
        print("Invalid tag")
        return
    try:
        import shutil 
        shutil.rmtree(args.tag)
    except:
        pass
    os.mkdir(args.tag)

    global DATA_PATH_TRAIN, DATA_PATH_VALID, DATA_PATH_TEST
    DATA_PATH_TRAIN = args.tag+'/train/'
    DATA_PATH_VALID = args.tag+'/valid/'
    DATA_PATH_TEST = args.tag+'/test/'

    category = categorization()
    type_ = type__()
    # if search for blue tag, positive data is blue and negative data is white, red ...
    negative = type_[[k for k, v in type_.items() if args.tag in v][0]]
    negative.remove(args.tag)

    # read csv
    import pandas as pd
    df = pd.read_csv("Metadata.csv", encoding = "ISO-8859-15", header=None, low_memory=False)

    error = 0
    imgs_list = {"y":[],"n":[]}
    for _, row in df.iterrows():
        try:
            tags = row[5].strip().split('   ')
        except:
            continue
        # check if image is in ui tag
        if len([value for value in tags if value in category['ui']])==0:
            continue
        # check if image is in positive tag
        if len([value for value in tags if value in category[args.tag]])>0:
            imgs_list['y'].append(str(row[0]))
        # check if image is in negative tag
        elif len([x for x in tags for y in negative if x in category[y]])>0:
            imgs_list['n'].append(str(row[0]))
        else:
            continue
    # balance positive and negative
    imgs_list['n'] = random.choices(imgs_list['n'],k=len(imgs_list['y']))
    imgs_list['all'] = imgs_list['y'] + imgs_list['n']
    random.shuffle(imgs_list['all'])
    random.shuffle(imgs_list['all'])
    # random split dataset 0.8/0.1/0.1
    ratio_train = 0.8
    ratio_validation = 0.1
    p1 = int(ratio_train * len(imgs_list['all']))
    p2 = int(ratio_validation * len(imgs_list['all']))
    train = imgs_list['all'][:p1]
    valid = imgs_list['all'][p1:p1+p2]
    test = imgs_list['all'][p1+p2:]
    print('Train:', len(train))
    print('Valid:', len(valid))
    print('Test:', len(test))
    paths = {DATA_PATH_TRAIN:train, DATA_PATH_VALID:valid, DATA_PATH_TEST:test}
    for k,v in paths.items():
        os.mkdir(k)
        os.mkdir(k+'y')
        os.mkdir(k+'n')
        for p in v:
            # find image
            if os.path.exists(FROM+p+'.jpg'):
                p = p+'.jpg'
            elif os.path.exists(FROM+p+'.png'):
                p = p+'.png'
            else:
                continue
            try:
                # img = cv2.imread(FROM+p)
                img = Image.open(FROM+p)
                img = img.convert("RGB")
                # if len(img.getbands()) == 4:
                #     background = Image.new("RGB", img.size, (255, 255, 255))
                #     background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
                #     img = background.copy()
                if len(img.getbands())!=3:
                    print(img.getbands())
                # resize option
                if args.resize:
                #     img = cv2.resize(img, (args.resize_rate,args.resize_rate), interpolation=cv2.INTER_CUBIC)
                    img = img.resize((args.resize_rate,args.resize_rate))
                # save image
                if p.split('.')[0] in imgs_list['y']:
                    img.save(k+'y/'+p,quality=80)
                    # cv2.imwrite(k+'y/'+p)
                else:
                    img.save(k+'n/'+p,quality=80)
                    # cv2.imwrite(k+'n/'+p)
            except:
                error+=1
                continue   
    print("Error:",error)
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
            cv2.imwrite(save_dir_y + 'noise_{}.jpg'.format(j), img_out)
            j += 1
        except:
            pass

    for i in tqdm.tqdm(os.listdir(path_n)):
        try:
            img = cv2.imread(path_n + i)
            img_out = add_gasuss_noise(img)
            cv2.imwrite(save_dir_n + 'noise_{}.jpg'.format(j), img_out)
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