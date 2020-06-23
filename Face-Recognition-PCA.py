import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

#Loading the dataset
data1=fetch_lfw_people(min_faces_per_person=100)
n, h, w = data1.images.shape
print(data1.images.shape)
print(data1.data.shape)
print(data1.target.shape)
x=data1.data
y=data1.target
target_Names=data1.target_names

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#Compute PCA
n_components =100
pca= PCA(n_components=n_components,whiten=True).fit(x_train)

#Apply PCA
x_train_pca=pca.transform(x_train)
x_test_pca=pca.transform(x_test)
clf=MLPClassifier(hidden_layer_sizes=1024,batch_size=256,verbose=True, early_stopping=True).fit(x_train_pca,y_train)

y_pred=clf.predict(x_test_pca)
print(classification_report(y_test,y_pred,target_names=target_Names))


def plot_gallery(images, titles, h, w, rows=3, cols=4):
    plt.figure()
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())


def titles(y_pred, y_test, target_names):
    for i in range(y_pred.shape[0]):
        pred_name = target_names[y_pred[i]].split(' ')[-1]
        true_name = target_names[y_test[i]].split(' ')[-1]
        yield 'predicted: {0}\ntrue: {1}'.format(pred_name, true_name)


prediction_titles = list(titles(y_pred, y_test, target_Names))
plot_gallery(x_test, prediction_titles, h, w)