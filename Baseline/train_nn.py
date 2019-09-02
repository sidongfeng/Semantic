import argparse
import pickle
import torch
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from loader import data_loader
import pandas as pd

parser = argparse.ArgumentParser(description='Baseline')
parser.add_argument('--tag', type=str, default='blue',
                    help='model to train (default: blue)')
parser.add_argument('--classifer', type=str, default='svm',
                    help='binary classifer [svm, dt] (default: svm)')
parser.add_argument('--feature', type=str, default='hsv',
                    help='image feature [hist, hsv, sift] (default: hsv)')

args = parser.parse_args()
print(args)

df_train, df_test = data_loader(args.tag,args.feature)
# df = df_train.append(df_test, ignore_index = True) 
# f = open("test.csv","w")
# f.write(df.to_csv(index=False))
# f.close()
df_train = shuffle(df_train)


def train():
    X_train = df_train.iloc[:,1:]
    y_train = df_train.iloc[:,0]
    # # normalise input data
    # for column in X_train:
    #     norm = lambda x: (x - x.mean()) / x.std()
    #     print(norm)
    #     X_train[column] = X_train.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())
    X = torch.Tensor(X_train.values).float()
    Y = torch.Tensor(y_train.values).long()

   # define a customised neural network structure
    class TwoLayerNet(torch.nn.Module):

        def __init__(self):
            super(TwoLayerNet, self).__init__()
            # define linear hidden layer output
            self.hidden = torch.nn.Linear(50, 20)
            # define linear output layer output
            self.out = torch.nn.Linear(20, 2)

        def forward(self, x):
            """
                In the forward function we define the process of performing
                forward pass, that is to accept a Variable of input
                data, x, and return a Variable of output data, y_pred.
            """
            # get hidden layer input
            h_input = self.hidden(x)
            # define activation function for hidden layer
            h_output = torch.sigmoid(h_input)
            # get output layer output
            y_pred = self.out(h_output)

            return y_pred

    learning_rate = 0.1
    num_epoch = 5000
    net = TwoLayerNet()
    loss_func = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

    all_losses = []
    for epoch in range(num_epoch):
        Y_pred = net(X)
        loss = loss_func(Y_pred, Y)
        all_losses.append(loss.item())

        if epoch % 50 == 0:
            _, predicted = torch.max(Y_pred, 1)
            total = predicted.size(0)
            correct = predicted.data.numpy() == Y.data.numpy()
            print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
                % (epoch + 1, num_epoch, loss.item(), 100 * sum(correct)/total))

        net.zero_grad()
        loss.backward()
        optimiser.step()
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(all_losses)
    plt.show()

    X_test = df_test.iloc[:,1:-1]
    y_test = pd.to_numeric(df_test.iloc[:,0])
    X = torch.Tensor(X_test.values).float()
    Y = torch.Tensor(y_test.values).long()

    Y_pred_test = net(X)
    _, predicted_test = torch.max(Y_pred_test, 1)
    total_test = predicted_test.size(0)
    correct_test = sum(predicted_test.data.numpy() == Y.data.numpy())

    print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))

def train_svm():
    from sklearn.model_selection import train_test_split
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Standardize
    # stdScale = StandardScaler().fit(X_train)
    # X_train = stdScale.transform(X_train)
    # X_test = stdScale.transform(X_test)
    from sklearn.svm import SVC
    svclassifier = SVC(kernel='linear')
    print('-'*10)
    print(svclassifier)
    svclassifier.fit(X_train, y_train)
    with open('model/svm.pickle', 'wb') as f:
        pickle.dump(svclassifier, f)
    y_pred = svclassifier.predict(X_test)
    print(y_pred)
    print('-'*10)
    print(confusion_matrix(y_test,y_pred))
    print("Accuarcy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test,y_pred,labels=[0,1]))



if __name__ == "__main__":
    train()
    # train_svm()
    # test()