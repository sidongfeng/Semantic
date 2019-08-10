import sys
import csv
import tqdm
import pandas as pd
import numpy as np
sys.path.append('../Data/')
import categorization

WORD2VEC = "../Data/glove.6B.50d.txt"
FILE = "./table.csv"

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

# loading categorization
def loadCategory():
    category = categorization.categorization()
    return category

# preprocess Metadata.csv
def load_image_tags(f=FILE):
    print('-'*10)
    print("Preprocessing Image_tag.....")
    print('-'*10)

    # preprocess Metadata.csv
    df = pd.read_csv(f, header=None)
    # print(df.loc[df[0]== 4000000])
    category = loadCategory()
    
    img_tags = {"origin":{}, "direct":{}}
    for _, row in tqdm.tqdm(df.iterrows()):
        img, string = row[0], row[1]
        try:
            string = string.split('+')
            string = [x for x in string if x not in category['ui']]
            tags = [k for k,v in category.items() if len(intersection(string,v))>0]
            string = [x.replace(' ','') for x in string]
            string = list(set(string))
            string = [x for x in string if len(x) > 1]
            string = [x for x in string if not x.isdigit()]
            string = [x.lower() for x in string]
            string.sort(key = len)
        except:
            string = []
            tags = []
        img_tags["origin"][img] = string
        img_tags["direct"][img] = tags
    return img_tags

# load Word2Vec model
def loadGloveModel(gloveFile=WORD2VEC):
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

def initial_csv():
    df = pd.read_csv(FILE)
    return df

def update_csv(img_ids,t,df):
    for id in img_ids:
        predict_tags = df.loc[df["id"]== id,"predict"].item()
        if predict_tags == " ":
            df.loc[df["id"]== id,"predict"] = t
        else:
            df.loc[df["id"]== id,"predict"] = predict_tags + '+' + t
    return df

def write_csv(df):
    fo = open("result.csv","w")
    fo.write(df.to_csv(index=False))
    fo.close()

if __name__ == "__main__":
    # loading dictionary of tags for each image, regardless the related tags
    # print(load_image_tags()["direct"])

    # loading a dictionary for Glove
    # loadGloveModel()

    # a = initial_csv()
    # a = update_csv([3919542],"red",a)
    # write_csv(a)
    None