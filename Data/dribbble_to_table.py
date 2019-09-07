import os
from shutil import copyfile
import csv

def not_plat():
    os.chdir("dribbble")

    img_list = {}

    categories = os.listdir("./")
    for cat in categories:
        if cat == ".DS_Store":
            continue
        os.chdir(cat)

        tags = os.listdir("./")
        for tag in tags:
            if tag == ".DS_Store":
                continue
            os.chdir(tag)
            imgs = os.listdir()
            for img in imgs:
                if img == ".DS_Store":
                    continue
                if img not in img_list.keys():
                    img_list[img] = tag
                else:
                    img_list[img] += '+'+tag
                src = img
                dst = '/Users/mac/Documents/Python/Semantic/Data/images/'+img
                copyfile(src,dst)
            os.chdir('../')

        os.chdir('../')
    print(img_list)

    with open('table.csv', 'w') as f:
        for key in img_list.keys():
            f.write("%s,%s\n"%(key,img_list[key]))
    f.close()

def test():
    import pandas as pd
    df = pd.read_csv('table.csv',header=None)
    
    imgs = os.listdir('platform')
    try:
        imgs.remove('.DS_Store')
    except:
        pass
    for img in imgs:
        if img.startswith('1_'):
            if len(df[df[0]==img]) == 0:
                df = df.append({0: img, 1:'mobile'}, ignore_index=True)
            else:
                df.loc[df[0] == img, [1]] = df.loc[df[0] == img, [1]]+'+mobile'
        else:
            if len(df[df[0]==img]) == 0:
                df = df.append({0: img, 1:'website'}, ignore_index=True)
            else:
                df.loc[df[0] == img, [1]] = df.loc[df[0] == img, [1]]+'+website'

    f = open('table1.csv','w')
    f.write(df.to_csv(index=False))
    f.close()




if __name__ == "__main__":
    # generate not platform images
    test()