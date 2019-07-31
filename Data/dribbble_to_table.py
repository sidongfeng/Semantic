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





if __name__ == "__main__":
    # generate not platform images
    not_plat()