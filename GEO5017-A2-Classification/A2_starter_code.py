"""
This demo shows how to visualize the designed features. Currently, only 2D feature space visualization is supported.
I use the same data for A2 as my input.
Each .xyz file is initialized as one urban object, from where a feature vector is computed.
6 features are defined to describe an urban object.
Required libraries: numpy, scipy, scikit learn, matplotlib, tqdm 
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.spatial import ConvexHull
from tqdm import tqdm
from os.path import exists, join
from os import listdir
from sklearn.ensemble import RandomForestClassifier


class urban_object:
    """
    Define an urban object
    """
    def __init__(self, filenm):
        """
        Initialize the object
        """
        # obtain the cloud name
        self.cloud_name = filenm.split('/\\')[-1][-7:-4]

        # obtain the cloud ID
        self.cloud_ID = int(self.cloud_name)

        # obtain the label
        self.label = math.floor(1.0*self.cloud_ID/100)

        # obtain the points
        self.points = read_xyz(filenm)

        # initialize the feature vector
        self.feature = []

    def compute_features(self):
        """
        Computing the features for the pointcloud dataset

        Features taken into account :
            - Linearity
            - Sphericity
            - Planarity
            - Verticality
            - Area
            - Relative Z height
        """
        kd_tree_3d = KDTree(self.points, leaf_size=5)
        
        k_top = max(int(len(self.points) * 0.005), 100)

        top = self.points[[np.argmax(self.points[:, 2])]]

        idx = kd_tree_3d.query(top, k=k_top, return_distance=False)

        idx = np.squeeze(idx, axis=0)

        neighbours = self.points[idx, :]

        cov = np.cov(neighbours.T)

        w, v = np.linalg.eig(cov)

        w.sort()

        #Feature 1 : Linearity
        linearity = (w[2]-w[1]) / (w[2] + 1e-5)

        #Feature 2 : Sphericity
        sphericity = w[0] / (w[2] + 1e-5)

        #Feature 3 : Planarity
        planarity = (w[1] - w[0] )/( w[2] + 1e-5)

        #Feature 4: Verticality
        idx = w.argsort()[::-1] 
        v = v[:, idx]
        e1, e2, e3 = v[:, 0], v[:, 1], v[:, 2]  # noqa: F841
        z_axis = np.array([0, 0, 1])
        verticality = 1 - np.abs(np.dot(e3, z_axis))

        #Featue 5: Area 
        hull_2d = ConvexHull(self.points[:, :2])
        hull_area = hull_2d.volume

        #Feature 6: Relative Z height
        height_max = np.amax(self.points[:, 2])
        height_min = np.amin(self.points[:, 2])

        relative_height = height_max-height_min

        self.feature += [linearity, sphericity, planarity, verticality, hull_area, relative_height]


def feature_selection(features, labels):
    """
        Selects best features from among all given features
        Returns only the best four features from among the given input

        Input: 
            - Features
            - Labels
        
        Output:
        Best Four Features

    """
    num_classes = 5
    num_samples = len(features)
    num_features = len(features[0])

    # calculate within class scatter matrix SW per feature
    sws = np.array([])
    for f in range(num_features):
        for c in range(num_classes):
            num_samples_class = np.sum(labels==c)
            feature_cov = np.cov(features[labels==c,f])
            if f == 0:
                sw = np.zeros_like(feature_cov)
            sw += (float(num_samples_class) / num_samples * feature_cov)
        #sw = np.trace(sw)
        sws = np.append(sws, sw)
    
    # calculate between class scatter matrix SB per feature
    sbs = np.array([])
    for f in range(num_features):
        mean_global = np.mean(features[:,f])
        for c in range(num_classes):
            num_samples_class = np.sum(labels==c)
            mean_class = np.mean(features[labels==c,f])
            if f == 0:
                sb = 0
            sb += (float(num_samples_class) / num_samples * (mean_class - mean_global) * (mean_class - mean_global))
        #sb = np.trace(sb)
        sbs = np.append(sbs, sb)

    # calculate J and keep only best four features based on it
    js = sbs / sws
    best_features = np.argpartition(js, -4)[-4:]
    best_features = np.sort(best_features)
    return features[:,best_features],best_features

def normalise_features(features):
    """
        Normalises the features
        Returns the normalised features
    """
    num_features = len(features[0])
    for f in range(num_features):
        feature = features[:,f]
        norm_feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
        features[:, f] = norm_feature
    return features

def read_xyz(filenm):
    """
    Reading points
        filenm: the file name
    """
    points = []
    with open(filenm, 'r') as f_input:
        for line in f_input:
            p = line.split()
            p = [float(i) for i in p]
            points.append(p)
    points = np.array(points).astype(np.float32)
    return points


def feature_preparation(data_path):
    """
    Prepare features of the input point cloud objects
    data_path: the path to read data
    """
    # check if the current data file exist
    data_file = 'data.txt'
    if exists(data_file):
        return

    # obtain the files in the folder
    files = sorted(listdir(data_path))

    # initialize the data
    input_data = []

    # retrieve each data object and obtain the feature vector
    for file_i in tqdm(files, total=len(files)):
        # obtain the file name
        file_name = join(data_path, file_i)

        # read data
        i_object = urban_object(filenm=file_name)

        # calculate features
        i_object.compute_features()

        # add the data to the list
        i_data = [i_object.cloud_ID, i_object.label] + i_object.feature
        input_data += [i_data]

    # transform the output data
    outputs = np.array(input_data).astype(np.float32)

    # write the output to a local file
    data_header = 'ID,label,Linearity,Sphericity,Planarity,Verticality,Hull Area,Relative Height'
    np.savetxt(data_file, outputs, fmt='%10.5f', delimiter=',', newline='\n', header=data_header)


def data_loading(data_file='data.txt'):
    """
    Read the data with features from the data file
        data_file: the local file to read data with features and labels
    """
    # load data
    data = np.loadtxt(data_file, dtype=np.float32, delimiter=',', comments='#')

    # extract object ID, feature X and label Y
    ID = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    X = data[:, 2:].astype(np.float32)

    return ID, X, y


def feature_visualization(bf,X):
    """
    Visualize the features
        X: input features. This assumes classes are stored in a sequential manner
    """
    # initialize a plot
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title("Feature subset visualization of 5 classes", fontsize="small")

    # define the labels and corresponding colors
    colors = ['firebrick', 'grey', 'darkorange', 'dodgerblue', 'olivedrab']
    labels = ['building', 'car', 'fence', 'pole', 'tree']

    print(np.shape(X))
    # plot the data with first two features
    for i in range(5):
        ax.scatter(X[100*i:100*(i+1), 2], X[100*i:100*(i+1), 3], marker="o", c=colors[i], edgecolor="k", label=labels[i])

    # show the figure with labels

    feature_names={
        0:"Linearity",
        1:"Sphericity",
        2:"Planarity",
        3:"Verticality",
        4:"Area",
        5:"Relative Height"
    }

    a=feature_names[bf[0]]
    b=feature_names[bf[1]]

    """
    Replace the axis labels with your own feature names
    """
    ax.set_xlabel(f'x1: {a}')
    ax.set_ylabel(f'x2: {b}')
    ax.legend()
    plt.show()

def confusion_matrix_display(flag,classifier,X_test,y_test):

    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    if flag==0:
        titles_options = [
            ("SVM Confusion Matrix: Non-Normalised", None),
            ("SVM Confusion Matrix: Normalised", "true"),
        ]
    
    else:
        titles_options = [
            ("RF Confusion Matrix: Non-Normalised", None),
            ("RF Confusion matrix: Normalised", "true"),
        ]


    class_names=["Building","Car","Fence","Pole","Tree"]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            classifier,
            X_test,
            y_test,
            display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)

        #print(title)
        #print(disp.confusion_matrix)

    plt.show()



def SVM_classification(X, y):
    """
    Conduct SVM classification
        X: features
        y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_preds = clf.predict(X_test)
    acc = accuracy_score(y_test, y_preds)
    print("SVM accuracy: %5.2f" % acc)
    #print("confusion matrix")
    #conf = confusion_matrix(y_test, y_preds)
    #print(conf)
    print("Visualise the Confusion Matrix")
    flag=0
    confusion_matrix_display(flag=flag,classifier=clf,X_test=X_test,y_test=y_test)


def RF_classification(X, y):
    """
    Conduct RF classification
        X: features
        y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_preds = clf.predict(X_test)
    acc = accuracy_score(y_test, y_preds)
    print("Random Forest accuracy: %5.2f" % acc)
    print("Visualise the Confusion Matrix : Random Forest")
    flag=1
    confusion_matrix_display(flag=flag,classifier=clf,X_test=X_test,y_test=y_test)

def learning_curve(X,y):

    test_size_split=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    sample_size=[]
    svm_acc=[]
    rf_acc=[]

    for i in test_size_split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
        sample_size.append(len(X_train))
        clf_svm = svm.SVC()
        clf_rf = RandomForestClassifier()
        clf_svm.fit(X_train, y_train)
        clf_rf.fit(X_train, y_train)
        y_preds_svm = clf_svm.predict(X_test)
        acc = accuracy_score(y_test, y_preds_svm)
        svm_acc.append(acc)
        y_preds_rf = clf_rf.predict(X_test)
        acc = accuracy_score(y_test, y_preds_rf)
        rf_acc.append(acc)


    
    fig, ax = plt.subplots()

    ax.plot(sample_size, svm_acc, marker='o', label='SVM')
    ax.plot(sample_size, rf_acc, marker='s', label='Random Forest')

    ax.set_title("Learning Curve")
    ax.set_xlabel("Training Sample Size")
    ax.set_ylabel("Accuracy")

    #Accuracy Range
    ax.set_ylim(0, 1) 
    ax.grid(True)
    ax.legend()

    plt.show()
    


if __name__=='__main__':
    # specify the data folder
    """"Here you need to specify your own path"""
    path = 'GEO5017-A2-Classification/pointclouds-500'

    # conduct feature preparation
    print('Start preparing features')
    feature_preparation(data_path=path)

    # load the data
    print('Start loading data from the local file')
    ID, X, y = data_loading()

    # normalise data
    print("Normalising data")
    X = normalise_features(X)

    # select features
    print("Selecting features")
    X , best_features= feature_selection(X, y)

    # visualize features
    print('Visualize the features')
    feature_visualization(best_features,X=X)

    # SVM classification
    print('Start SVM classification')
    SVM_classification(X, y)

    # RF classification
    print('Start RF classification')
    RF_classification(X, y)
    learning_curve(X,y)


