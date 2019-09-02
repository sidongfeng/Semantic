import argparse
import pickle
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from loader import data_loader

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
df_train = shuffle(df_train)

def train():
    X_train = df_train.drop('label', axis=1)
    y_train = df_train['label']
    X_test = df_test.drop(['label','name'], axis=1)
    y_test = df_test['label'].tolist()
    print(y_test)
    
    # Standardize
    stdScale = StandardScaler().fit(X_train)
    X_train = stdScale.transform(X_train)
    X_test = stdScale.transform(X_test)

    if args.classifer == 'svm':
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
    else:
        from sklearn import tree
        dtclassifer = tree.DecisionTreeClassifier()
        print('-'*10)
        print(dtclassifer)
        dtclassifer.fit(X_train, y_train)
        y_pred = dtclassifer.predict(X_test)
        print('-'*10)
        print(confusion_matrix(y_test,y_pred))
        print("Accuarcy: ", accuracy_score(y_test, y_pred))
        print(classification_report(y_test,y_pred))
        # import graphviz 
        # dot_data = tree.export_graphviz(clf, out_file=None) 
        # graph = graphviz.Source(dot_data) 
        # graph.render("iris") 
        # tree.plot_tree(dtclassifer.fit(X_train, y_train)) 
        # dot_data = tree.export_graphviz(clf, out_file=None, 
        #               feature_names=iris.feature_names,  
        #               class_names=iris.target_names,  
        #               filled=True, rounded=True,  
        #               special_characters=True)  
        # graph = graphviz.Source(dot_data)  
        # graph

def test():
    with open('model/svm.pickle', 'rb') as f:
        svclassifier = pickle.load(f)

    X_train = df_train.drop('label', axis=1)
    X_test = df_test.drop(['label','name'], axis=1)

    y_test = df_test['label']
    y_name = df_test['name']

    stdScale = StandardScaler().fit(X_train)
    X_test = stdScale.transform(X_test)
    y_pred = list(svclassifier.predict(X_test))

    from shutil import copyfile
    right = [y_pred.index(x) for x in y_pred if x == 0]
    wrong = [y_pred.index(x) for x in y_pred if x == 1]
    right = []
    for i,x in enumerate(y_pred):
        if x==0:
            right.append(i)

    wrong = []
    for i,x in enumerate(y_pred):
        if x==1:
            wrong.append(i)

    for i in wrong:
        try:
            copyfile("/Users/mac/Documents/Python/Semantic/Data/"+args.tag+"/test/y/"+y_name[i], "/Users/mac/Documents/Python/Semantic/Baseline/test/right/"+y_name[i])
        except:
            copyfile("/Users/mac/Documents/Python/Semantic/Data/"+args.tag+"/test/n/"+y_name[i], "/Users/mac/Documents/Python/Semantic/Baseline/test/right/"+y_name[i])
    for i in right:
        try:
            copyfile("/Users/mac/Documents/Python/Semantic/Data/"+args.tag+"/test/y/"+y_name[i], "/Users/mac/Documents/Python/Semantic/Baseline/test/wrong/"+y_name[i])
        except:
            copyfile("/Users/mac/Documents/Python/Semantic/Data/"+args.tag+"/test/n/"+y_name[i], "/Users/mac/Documents/Python/Semantic/Baseline/test/wrong/"+y_name[i])


if __name__ == "__main__":
    train()
    test()