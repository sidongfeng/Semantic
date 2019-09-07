import os
import torch
from shutil import copyfile, rmtree
import model
from loader import image_generator
from loader import tag_generator
from function import load_image_tags
from function import loadGloveModel
from function import loadCategory
from function import intersection
from function import initial_csv
from function import update_csv
from function import write_csv

THRESHOLD = 0.85

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_types = ['platform','form','chart','grid','list_','dashboard','profile','checkout','landing', 'weather','sport','game','finance','travel','food','ecommerce','music','pink','black','white','green','blue','red','yellow']

# data loader
img_tags = load_image_tags()
w2v = loadGloveModel()
test_loader = image_generator()

model_image, _ = model.initialize_model("resnet", feature_extract=True, use_pretrained=True)
model_tag = model.CNN()
model = model.MyEnsemble(model_image, model_tag)
model = model.to(device)

def test_model(tag):
    if device==  torch.device('cpu'):
        model.load_state_dict(torch.load('backup/'+tag+'/checkpoint.pt',map_location='cpu'))
    else:
        model.load_state_dict(torch.load('backup/'+tag+'/checkpoint.pt'))
    model.eval()

    for img_inputs, _, paths in test_loader:
        img_inputs = img_inputs.to(device)
        tag_inputs = tag_generator(paths, img_tags["origin"], w2v, tag).to(device)
    
    outputs = torch.sigmoid(model(img_inputs, tag_inputs))
    for i in range(len(outputs)):
        if outputs[i][1]<=THRESHOLD:
            outputs[i][1]=0
    _, preds = torch.max(outputs, 1)

    if t != 'platform':
        model_based_id = [int(paths[i].split('/')[-1].split('.')[0]) for i in range(len(preds)) if int(preds[i]) == 1]
        return model_based_id
    else:
        mobile_model_based_id = [int(paths[i].split('/')[-1].split('.')[0]) for i in range(len(preds)) if int(preds[i]) == 0]
        web_model_based_id = [int(paths[i].split('/')[-1].split('.')[0]) for i in range(len(preds)) if int(preds[i]) == 1]
        return mobile_model_based_id,web_model_based_id

if __name__ == "__main__":
    RECOVERING_FOLDER = "recovering/"
    category = loadCategory()
    df = initial_csv()

    try:
        rmtree(RECOVERING_FOLDER)
    except:
        pass
    os.mkdir(RECOVERING_FOLDER)
    for t in model_types:
        print('-'*10)
        print('Processing '+ t+' .....')
        
        if t != 'platform':
            model_based_id = test_model(t)
            if t == 'list_': t = 'list'
            tag_based_id = [k for k, v in img_tags["direct"].items() if t in v]
            new_id = list(set(model_based_id) - set(intersection(model_based_id,tag_based_id)))
            union_id = model_based_id+tag_based_id
            union_id = set([int(x) for x in union_id])
            # tag_based_id = [int(x) for x in tag_based_id]
            # model_based_id = [int(x) for x in model_based_id]

            os.mkdir(RECOVERING_FOLDER+t)
            for i in new_id:
                src = 'images/1/'+str(i)+'.png'
                dst = RECOVERING_FOLDER+t+'/'+str(i)+'.png'
                copyfile(src,dst)
            print(len(new_id))

            df = update_csv(union_id, t, df)
        else:
            mobile_model_based_id,website_model_based_id = test_model(t)

            mobile_tag_based_id = [k for k, v in img_tags["direct"].items() if 'mobile' in v]
            mobile_new_id = list(set(mobile_model_based_id) - set(intersection(mobile_model_based_id,mobile_tag_based_id)))
            mobile_union_id = mobile_model_based_id+mobile_tag_based_id
            mobile_union_id = set([int(x) for x in mobile_union_id])
    
            website_tag_based_id = [k for k, v in img_tags["direct"].items() if 'website' in v]
            website_new_id = list(set(website_model_based_id) - set(intersection(website_model_based_id,website_tag_based_id))) 
            website_union_id = website_model_based_id+website_tag_based_id
            website_union_id = set([int(x) for x in website_union_id])
    
            os.mkdir(RECOVERING_FOLDER+'mobile')
            for i in mobile_new_id:
                src = 'images/1/'+str(i)+'.png'
                dst = RECOVERING_FOLDER+'mobile/'+str(i)+'.png'
                copyfile(src,dst)
            print('mobile: ', len(mobile_new_id))
            df = update_csv(mobile_union_id, 'mobile', df)
    
            os.mkdir(RECOVERING_FOLDER+'website')
            for i in website_new_id:
                src = 'images/1/'+str(i)+'.png'
                dst = RECOVERING_FOLDER+'website/'+str(i)+'.png'
                copyfile(src,dst)
            print('website: ', len(website_new_id))
            df = update_csv(website_union_id, 'website', df)

    write_csv(df)
