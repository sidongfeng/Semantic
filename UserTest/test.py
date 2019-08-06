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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_types = ['form','chart','grid','list_','dashboard','profile','checkout','landing', 'weather','sport','game','finance','travel','food','ecommerce','music','pink','black','white','green','blue','red','yellow']
model_types = ['blue']

# data loader
img_tags = load_image_tags()
w2v = loadGloveModel()
test_loader = image_generator()

model_image, _ = model.initialize_model("resnet", feature_extract=True, use_pretrained=True)
model_tag = model.CNN()
model = model.MyEnsemble(model_image, model_tag)
model = model.to(device)

def test_model(tag):
    if device == 'cpu':
        model.load_state_dict(torch.load('backup/'+tag+'/checkpoint.pt',map_location='cpu'))
    else:
        model.load_state_dict(torch.load('backup/'+tag+'/checkpoint.pt'))
    model.eval()

    for img_inputs, _, paths in test_loader:
        img_inputs = img_inputs.to(device)
        tag_inputs = tag_generator(paths, img_tags["origin"], w2v, tag).to(device)

    outputs = model(img_inputs, tag_inputs)
    _, preds = torch.max(outputs, 1)
    preds_id = [int(paths[i].split('/')[-1].split('.')[0]) for i in range(len(preds)) if int(preds[i]) == 1]
    return preds_id

if __name__ == "__main__":
    TAG = 'blue'
    RECOVERING_FOLDER = "recovering/"
    category = loadCategory()
    try:
        rmtree(RECOVERING_FOLDER)
    except:
        pass
    os.mkdir(RECOVERING_FOLDER)
    for t in model_types:
        print('-'*10)
        print('Processing '+ t+' .....')
        preds_id = test_model(t)
        dribbble_id = [k for k, v in img_tags["direct"].items() if t in v]
        union = list(set(preds_id+dribbble_id))
        inter = intersection(preds_id,dribbble_id)
        if len(dribbble_id)==0:
            print("Totally detect %i images, Coverage Ratio %f, Model-based %i images, Tag-based %i images"%(len(union), 0.0, len(preds_id)-len(inter), len(dribbble_id)))
        else:
            print("Totally detect %i images, Coverage Ratio %f, Model-based %i images, Tag-based %i images"%(len(union), len(inter)/len(dribbble_id), len(preds_id)-len(inter), len(dribbble_id)))
        os.mkdir(RECOVERING_FOLDER+t)
        # missing = [id for id in preds_id if len(intersection(category[t],img_tags[int(id)]))>0]
        print('-'*10)
        for i in set(preds_id)-set(inter):
            src = 'images/1/'+str(i)+'.png'
            dst = RECOVERING_FOLDER+t+'/'+str(i)+'.png'
            copyfile(src,dst)

        if t == TAG:
            try:
                rmtree("Check")
            except:
                pass
            os.mkdir("Check")
            for i in set(img_tags["direct"].keys()) - set(union):
                src = 'images/1/'+str(i)+'.png'
                dst = 'Check/'+str(i)+'.png'
                copyfile(src,dst)
