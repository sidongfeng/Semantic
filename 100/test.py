import sys, os
import tqdm
import pandas as pd
import shutil
import numpy as np

FILE = "../Data/Metadata.csv"
Image_Path = "../Data/images/"
# "iphone","ecommerce","checkout"
# mobile food drink listview
QUERY = ["iphone","ecommerce","checkout"]

def fullymatch():
    fo = open("table.csv",'a')
    try:
        shutil.rmtree("./images/")
    except:
        pass
    os.mkdir("./images/")
    no = 0

    df = pd.read_csv(FILE, encoding = "ISO-8859-15", header=None, low_memory=False)

    for _, row in df.iterrows():
        id,src,string = row[0],row[1],row[5]
        try:
            string = string.lower()
        except:
            continue
        string = string.strip().replace('   ','+')
        tags = string.split('+')
        if len(set(QUERY).intersection(tags)) == len(QUERY):
            fo.write(str(id)+","+string+','+src+"\n")
            # shutil.copyfile(Image_Path+str(id)+".png","./images/"+str(id)+".png")
            no+=1
    print(no)
    fo.close()

if __name__ == "__main__":    
    fullymatch()
