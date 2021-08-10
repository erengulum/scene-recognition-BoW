import numpy as np
from PIL import Image
import splitfolders
import os
from glob import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2 as cv
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score






def descr_list(data_dict):
    """ data_dict isimli sözlük'te, key olarak path, value olarak ise label'i yazılıdır.
    data_dict'te belirtilen bütün resimler tek tek açılır ve SIFT yöntemi  kullanılarak her resmin tek tek KP ve Descriptorları bulunur. """
    desc_list = []
    sift = cv.SIFT_create() # nfeatures= The number of best features to retain.
    for image_path, label in data_dict.items():
        im = cv.imread(image_path,0)
        kp = sift.detect(im, None)
        kp, descriptor = sift.compute(im, kp)
        if descriptor is not None:
            desc_list.append(descriptor)
    return desc_list





def create_vocab(desc_list,vocab_size):
    """ Görsellerin description'larının yer aldığı çok boyutlu liste ve vocabular size'ı input olarak alınır.
     KMeans sayesinde bu descriptionlar çok boyutlu uzayda kümelendirilir ve vocab oluşur. Return olarak bu vocab gönderilir"""
    descr_list_float = np.concatenate(desc_list, axis=0).astype('float32') #kmeans works only on float
    vocab = KMeans(n_clusters=vocab_size,random_state=0).fit(descr_list_float)
    return vocab





def build_histogram(desc_list, cluster_model):
    """ SIFT'den alınan description list ve daha önce oluşturulmuş olan vocabular(cluster_model) alınarak Histogramlar oluştur.
    Önce train kümesinin histogramları ardından test kümesinin histogramları oluşturulmalıdır. """
    hist_array = []
    for x in desc_list:
        histogram = np.zeros(len(cluster_model.cluster_centers_))
        cluster_result = cluster_model.predict(x.astype('float32'))
        for j in cluster_result:
            histogram[j] += 1.0
        hist_array.append(histogram)
    return hist_array





def drawCfMatrix(real_values, pred_values,matrixName):
    """Gerçek değerler, tahmin edilen değerler ve bir matrix ismi input olarak alınır.
    Bu verilerle, Confusion Matrix çizilerek verilen isimle aynı klasöre kaydedilir"""
    cm_array = confusion_matrix(real_values, pred_values)
    df_cm = pd.DataFrame(cm_array, range(6), range(6))
    sn.set(font_scale=1.0)
    ax = plt.axes()
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 20}, ax=ax)
    ax.set_title(matrixName)
    plt.savefig(matrixName)
    plt.clf()





def svm_classifier(train_features,train_label,test_features):
    """ It is basically a Linear Support Vector Classification function. Similar to SVC with parameter kernel=’linear’, but implemented
    in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and
    should scale better to large numbers of samples. Function takes train features, train label, test feature and returns predictions"""
    stdslr = StandardScaler().fit(train_features)
    train_features = stdslr.transform(train_features)
    test_features = stdslr.transform(test_features)
    """Linear SVC Parameters: Number of max_iters didn't change the success rates. class_weight='balanced' is increased 
    success rate of BOVW slightly but it decreased the sucess of Tiny Image Classification.So i decided to use default class_weight """
    classifier = LinearSVC(random_state=0, max_iter=1000, dual=False)
    classifier.fit(train_features, train_label)
    predicts = classifier.predict(test_features)
    return predicts





def knn_classifier(train_features,train_label,test_features,n_neig):
    """Function takes train features, train label, test feature, number of neighbours. It returns predictions made by model.
    Best results have been taken when n_neigbors set to 9"""
    knn = KNeighborsClassifier(n_neighbors=n_neig)
    knn.fit(train_features, train_label)
    predicts = knn.predict(test_features)
    return predicts





def split_train_test(data_path,d_output, subcategories,split_ratio):
    """Veri setinin path'i, oluşturulacak olan output klasörünün ismi ve verisetinin içerisinde bulunan subcategoriler input olarak alınır.
    splitfolders kütüphanesi sayesinde, veri seti belirtilen oranda train,validation ve test alt klasörlerine ayrılır. Seed parametresi sayesinde ise
    split işleminin random bir şekilde yapılması sağlanır. Oluşturulan bu yeni klasörler, loop sayesinde tek tek dolaştırılır ve bir sözlük yapisi oluşturulur
    Return olarak test_dict ve train_dict isimli iki sözlük yapısı dönderilir. """
    test_dict = {}
    train_dict = {}
    #dont use "/" at the beginning. The library will cut the slash at the beginning
    splitfolders.ratio(data_path, output=d_output, seed=1337, ratio=split_ratio)
    for category in subcategories:
        image_paths = glob(os.path.join(d_output, 'train', category, '*.jpg'))
        for i in range(len(image_paths)):
            train_dict[image_paths[i]] = category #image path ve karşılık label'i şeklinde bir dictionary yarattik

        image_paths = glob(os.path.join(d_output, 'test', category, '*.jpg'))
        for i in range(len(image_paths)):
            test_dict[image_paths[i]] = category #image path ve karşılık label'i şeklinde bir dictionary yarattik

    return train_dict, test_dict





def convert_to_tiny_img(data_dict):
    tiny_image_array = np.zeros((len(data_dict), 16 * 16))
    labels = [None] * len(data_dict)
    i=0
    for image_path,label in data_dict.items():
        im = Image.open(image_path)
        resized_im = np.asarray(im.resize((16, 16), Image.ANTIALIAS), dtype='float32').flatten()
        image_nm = (resized_im - np.mean(resized_im)) / np.std(resized_im)
        tiny_image_array[i] = image_nm
        labels[i] = label
        i = i+1

    return tiny_image_array, labels





def accuracy_f1score(name,true_values,predicts):
    """This function takes name of the model, true values and predictions and it prints Accuracy and F1_Score of the predicted values"""
    accuracy = accuracy_score(true_values, predicts)
    f1 = f1_score(true_values, predicts, average=None)
    print(name,"Accuracy:", accuracy, "and F1_Score:", f1)






#######################################   BEGIN   ###########################################################



"""   ################ SPLIT TEST AND TRAIN  ################ """
data_categories = ['Bedroom', 'Highway', 'Kitchen', 'LivingRoom', 'Mountain','Office']
train_dict, test_dict= split_train_test("SceneDataset", d_output="test_train_set",subcategories=data_categories, split_ratio=(0.7, 0,0.3)) #split_ratio=(train,validation,test)





"""   ################ TINY IMAGE #############################
 1- Convert training and test images to Tiny image
 2- Fit model by using training image and labels
 3- Predict the label of test set """
tiny_train_features, train_labels = convert_to_tiny_img(train_dict) #retrieve tiny train features and train labels from train_dict
tiny_test_features, test_labels = convert_to_tiny_img(test_dict)

# TINY IMAGE ML Models
tiny_knn_pred = knn_classifier(tiny_train_features, train_labels, tiny_test_features,n_neig=9) #function will return the predicted labels
tiny_svm_pred = svm_classifier(tiny_train_features, train_labels, tiny_test_features)

#TINY IMAGE ACCURACY AND F1 SCORES
accuracy_f1score("Tiny Image KNN",test_labels,tiny_knn_pred)
accuracy_f1score("Tiny Image SVM",test_labels,tiny_svm_pred)

#TINY IMAGE CONFUSION MATRIX
drawCfMatrix(test_labels, tiny_knn_pred,"Tiny Image - KNN Confusion Matrix")
drawCfMatrix(test_labels, tiny_svm_pred,"Tiny Image - SVM Confusion Matrix")





"""   ################ BAG OF VISUAL WORDS ############################# 
1- Define descriptions of train set by using SIFT
2- Create vocabulary by using descriptions and vocab_size
3- Build histograms of training set acc to Training Description list and created vocabulary
4- Build histograms of test set by using Test Set Descriptions and vocabulary
5- Fit the model with train histograms and labels
6- Predict the labels by using test histograms
"""

"""   Bag Of Visual Words - KNN Classifier     """


train_desc_list = descr_list(train_dict)
vocab = create_vocab(train_desc_list,vocab_size=45) #Vocabulary is created
train_hists = build_histogram(train_desc_list,vocab) #Histograms of the train set are created by using vocabulary

test_desc_list = descr_list(test_dict)
test_hists = build_histogram(test_desc_list,vocab)


#Predictions
bow_knn_predicts = knn_classifier(train_hists, train_labels, test_hists,n_neig=9)
bow_svc_predicts = svm_classifier(train_hists, train_labels,test_hists)

#BOVW ACCURACY AND F1 SCORES
accuracy_f1score("BoW KNN",test_labels,bow_knn_predicts)
accuracy_f1score("BoW SVM",test_labels,bow_svc_predicts)

#BOVW  CONFUSION MATRIX
drawCfMatrix(test_labels, bow_knn_predicts,"BoW - KNN Confusion Matrix")
drawCfMatrix(test_labels, bow_svc_predicts,"BoW - SVC Confusion Matrix")




"""
Actual labels(with their paths) vs predicted values 
for (key, value), predict in zip(test_dict.items(), bow_svc_predicts):
    print(key,value,"---",predict)
"""
