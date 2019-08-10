import sys, os, shutil
import tqdm
import pandas as pd
import torch.utils.data
import torchvision.transforms 
from torchvision import datasets
sys.path.append('../Data/')
import categorization

FILE = "./table.csv"
Image_Path = "./images/"

# loading categorization
def loadCategory():
    category = categorization.categorization()
    return category

def normalize(f=FILE):
    print('-'*10)
    print("normalizing Image_tag.....")
    print('-'*10)

    df = pd.read_csv(f, header=None)
    # print(df.loc[df[0]== 4000000])
    category = loadCategory()
    
    img_tags = {}
    for _, row in tqdm.tqdm(df.iterrows()):
        img, string = row[0], row[1]
        try:
            string = string.lower().split('+')
            tags = []
            for s in string:
                for k,v in category.items():
                    if s in v:
                        if k == 'list_':
                            tags.append('list')
                        else:
                            tags.append(k)
                    else:
                        tags.append(s)
            # tags += QUERY
            tags = list(set(tags))
        except:
            string = []
            tags = []
        img_tags[img] = (tags,string)
    
    f = open("table.csv","w")
    f.write("id,origin,normalize,predict\n")
    for k,v in img_tags.items():
        sstring = "+".join(v[1])
        tstring = "+".join(v[0])
        f.write(str(k)+','+sstring+","+tstring+", "+"\n")
    f.close()
    return img_tags

def remove_bad_format():
    df = pd.read_csv(FILE, header=None)

    os.chdir(Image_Path)
    imgs = os.listdir()
    try: 
        imgs.remove(".DS_Store")
    except:
        pass
    for i in imgs:
        id = i.split(".")[0]
        img_format = i.rsplit(".",1)[-1].lower()
        if img_format == "png":
            pass
        elif img_format == "jpg" or img_format == "jpeg":
            os.rename(i,id+".png")
        else:
            os.remove(i)
            df = df[df[0] != int(id)]
    
    os.chdir("../")
    fo = open("table.csv","w")
    fo.write(df.to_csv(index=False,header=None))
    fo.close()

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def remove_bad_image():
    os.mkdir('test')
    shutil.move(Image_Path,'test/images')

    df = pd.read_csv(FILE, header=None)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
    ])

    test_loader = ImageFolderWithPaths(root="test",transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_loader,batch_size=len(test_loader),shuffle=False)
    imgs = os.listdir("test/images/")
    good = []
    for i in tqdm.tqdm(range(len(test_loader.dataset))):
        try:
            path = test_loader.dataset[i][2]
            id = path.split("/")[-1].split('.')[0]
            good.append(id)
        except:
            pass
    for a in imgs:
        remove = True
        for b in good:
            if b in a:
                remove = False
        if remove: 
            print(a)
            os.remove('test/images/'+a)
            try:
                df = df[df[0] != int(a.split('.')[0])]
            except:
                pass

    fo = open("table.csv","w")
    fo.write(df.to_csv(index=False,header=None))
    fo.close()

    shutil.move('test/images',Image_Path)
    shutil.rmtree('test')

if __name__ == "__main__":
    remove_bad_format()
    remove_bad_image()
    normalize()
    
