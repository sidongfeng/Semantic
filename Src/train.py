import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from loader import data_generator
from earlystop import EarlyStopping
from model import initialize_model
# from BDR import BDR

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Recovering Missing Semantics')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (default: 0.2)')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit (default: 50)')
# parser.add_argument('--ksize', type=int, default=7,
#                     help='kernel size (default: 7)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 1e-4)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
# parser.add_argument('--hid', type=list, default=[10,10],
#                     help='number of hidden units per layer (default: [10,10])')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--patience', type=int, default=5,
                    help='number of patience to early stop (default: 5)')
# parser.add_argument('--backup', action='store_true',
#                     help='load Backup (default: false)')
# parser.add_argument('--bdr', action='store_false',
#                     help='use BDR (default: true)')
parser.add_argument('--early', action='store_false',
                    help='use Early Stop (default: true)')
# parser.add_argument('--distribution', action='store_true',
#                     help='visualise error distribution (default: false)')
# parser.add_argument('--process', action='store_true',
#                     help='visualise loss and accuracy (default: false)')

args = parser.parse_args()
# args = parser.parse_args(args=[])
torch.manual_seed(args.seed)
print(args)

# data loader
train_loader, valid_loader, test_loader = data_generator()

# Hyperparameters
epochs = args.epochs
learning_rate = args.lr
dropout = args.dropout
num_classes = 2
batch_size = 32
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
model = model.to(device)
# print(model)

# load pretrained model
# if args.backup:
#     if os.path.exists('backup/checkpoint.pt'):
#         model.load_state_dict(torch.load('backup/checkpoint.pt'))

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

optimizer = getattr(optim, args.optim)(params_to_update, lr=learning_rate, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# initialize the early_stopping object
early_stopping = EarlyStopping(args.patience)

#### train
def train_model():
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    print('train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc

#### valid
def valid_model():
    model.eval() # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(valid_loader.dataset)
    epoch_acc = running_corrects.double() / len(valid_loader.dataset)
    print('valid Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc

######################    
# test the model #
######################
def test_model():
    model.load_state_dict(torch.load('backup/checkpoint.pt'))
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / len(test_loader.dataset)
    print('valid Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


    #         # deep copy the model
    #         if phase == 'val' and epoch_acc > best_acc:
    #             best_acc = epoch_acc
    #             best_model_wts = copy.deepcopy(model.state_dict())
    #         if phase == 'val':
    #             val_acc_history.append(epoch_acc)

    #     print()

    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # # load best model weights
    # model.load_state_dict(best_model_wts)
    # return model, val_acc_history

        

if __name__ == "__main__":

    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0
    train_loss_history = []
    valid_loss_history = []
    train_acc_history = []
    valid_acc_history = []

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)

        train_loss, train_acc = train_model()
        valid_loss, valid_acc = valid_model()

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        train_acc_history.append(train_acc)
        valid_acc_history.append(valid_acc)

        if args.early:
            early_stopping(valid_acc, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    
    # visualise loss diagram and accuracy diagram
    plt.figure(1)
    plt.plot(train_loss_history)
    plt.plot(valid_loss_history)
    maxposs = valid_acc_history.index(max(valid_acc_history))+1 
    # plt.axvline(maxposs, linestyle='--', color='r')
    plt.gca().legend(('Train','Validation', 'Early Stopping Checkpoint'))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.figure(2)
    plt.plot(train_acc_history)
    plt.plot(valid_acc_history)
    maxposs = valid_acc_history.index(max(valid_acc_history))+1 
    # plt.axvline(maxposs, linestyle='--', color='r')
    plt.gca().legend(('Train','Validation','Early Stopping Checkpoint'))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy: %')
    plt.show()


    test_model()

    
    
#     if args.distribution:
#         # visualise errors distribution and BDR removal line
#         plt.figure(1)
#         plt.subplot(121)
#         mean_ts = np.mean(errors)
#         subset_outliers = [error for error in errors if error>1.5*mean_ts]
#         mean_ss = np.mean(np.array(subset_outliers))
#         std_ss = np.std(np.array(subset_outliers))
#         plt.title('Histogram of errors distribution')
#         plt.hist(errors, bins='auto')
#         plt.xlabel('Error')
#         plt.ylabel('Frequency')
#         plt.subplot(122)
#         plt.title('Errors Removal')
#         plt.scatter(range(len(errors)),errors,c='green',marker='.')
#         plt.plot(range(len(errors)),[mean_ss+1*std_ss] *len(errors), c='red',label='BDR')
#         plt.gca().legend()
#         plt.xlabel('Pattern')
#         plt.ylabel('Error')
#         plt.show()

#     if args.process:
#         # visualise loss diagram and accuracy diagram
#         plt.figure(2)
#         plt.subplot(121)
#         plt.plot(train_losses)
#         plt.plot(valid_losses)
#         maxposs = valid_accs.index(max(valid_accs))+1 
#         plt.axvline(maxposs, linestyle='--', color='r')
#         plt.gca().legend(('Train','Validation', 'Early Stopping Checkpoint'))
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.subplot(122)
#         plt.plot(train_accs)
#         plt.plot(valid_accs)
#         maxposs = valid_accs.index(max(valid_accs))+1 
#         plt.axvline(maxposs, linestyle='--', color='r')
#         plt.gca().legend(('Train','Validation','Early Stopping Checkpoint'))
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy: %')
#         plt.show()