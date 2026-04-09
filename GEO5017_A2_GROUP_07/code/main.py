import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.spatial import ConvexHull
from tqdm import tqdm
from os.path import exists, join
from os import listdir
from sklearn.ensemble import RandomForestClassifier

# specify the data folder
path = 'pointclouds-500'

# chosen_model = True sets classifiers hyperparameters to predefined ones:
# SVM, n_estimators: 100, max_depth: 5, max_features: "log 2"
# RF kernel: 'rbf', C: 10, Gamma: "scale"

# chosen_model = False lets the program find the best accuracy by iterating over multiple
# hyperparameters and selecting the best features for the training set.

chosen_model = True # Switch between predefined params and features (True) or optimising (False)

# Predefined hyperparameters
best_features = [6, 4, 7, 1]

# Predefined hyperparameters
chosen_params_rf = {
        'n_estimators': 100,
        'max_depth': 5,
        'max_features': 'log2'
    }
chosen_params_svm = {
        'kernel': 'rbf',
        'C': 10,
        "gamma": "scale"
    }

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
            - Planarity
            - Sphericity
            - Verticality
            - Area
            - Shape index
            - Relative Z height
            - Curvature change
        """
        kd_tree_3d = KDTree(self.points, leaf_size=5)
        
        k_top = max(int(len(self.points) * 0.005), 100)

        top = self.points[[np.argmax(self.points[:, 2])]]

        idx = kd_tree_3d.query(top, k=k_top, return_distance=False)

        idx = np.squeeze(idx, axis=0)

        neighbours = self.points[idx, :]

        cov = np.cov(neighbours.T)

        w, v = np.linalg.eig(cov)

        idx = w.argsort()[::-1]
        w = w[idx]
        v = v[:, idx]

        # Feature 1: Linearity
        linearity = (w[0] - w[1]) / (w[0] + 1e-5)

        # Feature 2: Planarity
        planarity = (w[1] - w[2]) / (w[0] + 1e-5)

        # Feature 3: Sphericity
        sphericity = w[2] / (w[0] + 1e-5)

        # Feature 4: Verticality
        e3 = v[:, 2]
        z_axis = np.array([0, 0, 1])
        verticality = 1 - np.abs(np.dot(e3, z_axis))

        #Featue 5: Area 
        hull_2d = ConvexHull(self.points[:, :2])
        hull_area = hull_2d.volume

        # Feature 6: Shape Index
        hull_perimeter = hull_2d.area
        shape_index = 1.0 * hull_area / hull_perimeter
        
        #Feature 7: Relative Z height
        height_max = np.amax(self.points[:, 2])
        height_min = np.amin(self.points[:, 2])

        relative_height = height_max-height_min

        #Feature 8: Curvature Change
        cur_change = w[0]/(w[0]+w[1]+w[2])


        self.feature += [linearity, sphericity, planarity, verticality, hull_area, shape_index,relative_height,cur_change]


def feature_selection(features, labels, d=4):
    """
    Forward selection using Fisher criterion:
    J = trace(SB) / trace(SW)

    Selects d features step by step and returns:
        - reduced feature matrix
        - selected feature indices
    """
    num_samples = features.shape[0]
    num_features = features.shape[1]
    classes = np.unique(labels)

    def fisher_score(feature_idx):
        """
        Compute J for a subset of features.
        """
        X = features[:, feature_idx]

        # ensure 2D shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        mean_global = np.mean(X, axis=0)

        SW = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
        SB = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)

        for c in classes:
            Xc = X[labels == c]
            Nc = Xc.shape[0]

            if Nc == 0:
                continue

            mean_class = np.mean(Xc, axis=0)

            # within-class scatter
            centered = Xc - mean_class
            SW += (centered.T @ centered) / num_samples

            # between-class scatter
            mean_diff = (mean_class - mean_global).reshape(-1, 1)
            SB += (Nc / num_samples) * (mean_diff @ mean_diff.T)

        return np.trace(SB) / (np.trace(SW) + 1e-12)

    selected_features = []
    remaining_features = list(range(num_features))

    while len(selected_features) < d:
        best_feature = None
        best_score = -np.inf

        for f in remaining_features:
            candidate_subset = selected_features + [f]
            score = fisher_score(candidate_subset)

            print(f"Testing subset {candidate_subset} -> J = {score:.6f}")

            if score > best_score:
                best_score = score
                best_feature = f

        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

        print(
            f"Selected feature {best_feature}, "
            f"current subset = {selected_features}, "
            f"J = {best_score:.6f}\n"
        )

    selected_features = np.array(selected_features)
    return features[:, selected_features], selected_features

def normalise_features(X_train, X_test):
    """
    Normalises the features
    """
    min_vals = np.min(X_train, axis=0)
    max_vals = np.max(X_train, axis=0)
    denom = (max_vals - min_vals) + 1e-12

    X_train_norm = (X_train - min_vals) / denom

    if X_test is not None:
        X_test_norm = (X_test - min_vals) / denom
        return X_train_norm, X_test_norm
    else:
        return X_train_norm


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
    data_header = 'ID,label,Linearity,Sphericity,Planarity,Verticality,Hull Area,Shape index,Relative Height,Curvature Change'
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


def feature_visualization(bf, X, y):
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
        mask = (y == i)
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            marker="o",
            c=colors[i],
            edgecolor="k",
            label=labels[i]
        )

    # show the figure with labels
    feature_names={
        0:"Linearity",
        1:"Sphericity",
        2:"Planarity",
        3:"Verticality",
        4:"Area",
        5:"Shape Index",
        6:"Relative Height",
        7:"Curvature Change"
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



def SVM_classification(X_train, X_test, y_train, y_test, chosen_model=False, chosen_params=None):
    """
    Conduct SVM classification with kernel and hyperparameter comparison
    on a precomputed train/test split.
    """
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    C_values = [0.01, 0.1, 1, 10]
    gamma_values = ['scale', 0.01, 0.1, 1]
    degree_values = [2, 3, 4, 5]

    results = []

    best_acc = -1
    best_params = None
    best_clf = None

    if chosen_model:
        best_params = chosen_params
        clf = svm.SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("\nChosen SVM params:", best_params)
        print(f"SVM accuracy: {acc:.5f}")
        return clf, results, best_params, acc

    for kernel in kernels:
        for C in C_values:

            if kernel == 'linear':
                clf = svm.SVC(kernel=kernel, C=C)

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                results.append((kernel, C, None, acc))
                print(f"SVM kernel={kernel:<8} C={C:<6} accuracy={acc:.5f}")

                if acc > best_acc:
                    best_acc = acc
                    best_params = {'kernel': kernel, 'C': C}
                    best_clf = clf

            elif kernel == 'poly':
                for gamma in gamma_values:
                    for degree in degree_values:
                        clf = svm.SVC(
                            kernel=kernel,
                            C=C,
                            gamma=gamma,
                            degree=degree
                        )

                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)

                        results.append((kernel, C, gamma, degree, acc))
                        print(
                            f"SVM kernel={kernel:<8} C={C:<6} "
                            f"gamma={str(gamma):<6} degree={degree:<2} "
                            f"accuracy={acc:.5f}"
                        )

                        if acc > best_acc:
                            best_acc = acc
                            best_params = {
                                'kernel': kernel,
                                'C': C,
                                'gamma': gamma,
                                'degree': degree
                            }
                            best_clf = clf
            else:
                for gamma in gamma_values:
                    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)

                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)

                    results.append((kernel, C, gamma, acc))
                    print(f"SVM kernel={kernel:<8} C={C:<6} gamma={str(gamma):<6} accuracy={acc:.5f}")

                    if acc > best_acc:
                        best_acc = acc
                        best_params = {'kernel': kernel, 'C': C, 'gamma': gamma}
                        best_clf = clf

    print("\nBest SVM params:", best_params)
    print(f"Best SVM accuracy: {best_acc:.5f}")

    return best_clf, results, best_params, best_acc


def RF_classification(X_train, X_test, y_train, y_test, chosen_model=False, chosen_params=None ):
    """
    Conduct RF classification with hyperparameter comparison
    on a precomputed train/test split.
    """
    n_estimators_values = [50, 100, 200]
    max_depth_values = [5, 10, 20, None]
    max_features_values = ['sqrt', 'log2']

    results = []

    best_acc = -1
    best_params = None
    best_clf = None

    if chosen_model:
        best_params = chosen_params
        clf = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], max_features=best_params['max_features'])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("\nChosen RF params:", best_params)
        print(f"RF accuracy: {acc:.5f}")

        return clf, results, best_params, acc


    for n_estimators in n_estimators_values:
        for max_depth in max_depth_values:
            for max_features in max_features_values:
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                )

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                results.append((n_estimators, max_depth, max_features, acc))

                print(
                    f"RF n_estimators={n_estimators:<3} "
                    f"max_depth={str(max_depth):<4} "
                    f"max_features={str(max_features):<5} "
                    f"accuracy={acc:.5f}"
                )

                if acc > best_acc:
                    best_acc = acc
                    best_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'max_features': max_features
                    }
                    best_clf = clf

    print("\nBest RF params:", best_params)
    print(f"Best RF accuracy: {best_acc:.5f}")

    return best_clf, results, best_params, best_acc

def learning_curve(X, y, best_features, svm_best_params, rf_best_params):
    """
    Learning curve for the final selected models.
    """

    test_size_split = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    sample_size = []
    svm_acc = []
    rf_acc = []
    svm_train_acc = []
    rf_train_acc = []

    for test_size in test_size_split:
        # split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
        )

        sample_size.append(len(X_train_raw))
        X_train, X_test = normalise_features(X_train_raw, X_test_raw)

        # apply fixed selected features
        X_train_sel = X_train[:, best_features]
        X_test_sel = X_test[:, best_features]

        # --- SVM with BEST params ---
        clf_svm = svm.SVC(**svm_best_params)

        # --- RF with BEST params ---
        clf_rf = RandomForestClassifier(**rf_best_params, random_state=42)

        # train
        clf_svm.fit(X_train_sel, y_train)
        clf_rf.fit(X_train_sel, y_train)
        y_pred_train = clf_svm.predict(X_train_sel)
        y_pred_test = clf_svm.predict(X_test_sel)
        svm_train_acc.append(accuracy_score(y_train, y_pred_train))
        svm_acc.append(accuracy_score(y_test, y_pred_test))
        y_pred_train = clf_rf.predict(X_train_sel)
        y_pred_test = clf_rf.predict(X_test_sel)
        rf_train_acc.append(accuracy_score(y_train, y_pred_train))
        rf_acc.append(accuracy_score(y_test, y_pred_test))

    # plot
    fig, ax = plt.subplots()

    # SVM
    ax.plot(sample_size, svm_train_acc, '--', marker='o', label='SVM train')
    ax.plot(sample_size, svm_acc, '-', marker='o', label='SVM test')

    # RF
    ax.plot(sample_size, rf_train_acc, '--', marker='s', label='RF train')
    ax.plot(sample_size, rf_acc, '-', marker='s', label='RF test')

    ax.set_title("Learning Curve")
    ax.set_xlabel("Training Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()

    plt.show()


if __name__=='__main__':
    class_names = ["building", "car", "fence", "pole", "tree"]
    feature_names = {
        0: "Linearity",
        1: "Sphericity",
        2: "Planarity",
        3: "Verticality",
        4: "Area",
        5: "Shape Index",
        6: "Relative Height",
        7: "Curvature Change"
    }

    # conduct feature preparation
    print('Start preparing features')
    feature_preparation(data_path=path)

    # load the data
    print('Start loading data from the local file')
    ID, X_raw, y = data_loading()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y,
        test_size=0.4,
        stratify=y
    )

    # normalise using only training data
    print("Normalising data")
    X_train, X_test = normalise_features(X_train_raw, X_test_raw)
    X_all = normalise_features(X_raw, None)

    # select features
    print("Selecting features")
    if chosen_model:
        X_train = X_train[:, best_features]
    else:
        X_train, best_features = feature_selection(X_train, y_train, d=4)
    X_test = X_test[:, best_features]

    selected_feature_names = [feature_names[i] for i in best_features]
    print(f"Selected feature names {best_features}:", selected_feature_names)


    # visualize features
    print('Visualize the features')
    feature_visualization(best_features,X_train,y_train)

    # SVM classification
    print('Start SVM classification')
    svm_clf, svm_results, svm_best_params, svm_best_acc = SVM_classification(
        X_train, X_test, y_train, y_test, chosen_model, chosen_params_svm
    )
    best_f1_svm = f1_score(y_test, svm_clf.predict(X_test), average=None, labels=[0,1,2,3,4])
    f1_svm = f1_score(y_test, svm_clf.predict(X_test), average='macro')
    print(f"Best SVM F1-score for each class: {class_names[0]}: {best_f1_svm[0]:.5f}, {class_names[1]}: {best_f1_svm[1]:.5f}, {class_names[2]}: {best_f1_svm[2]:.5f}, {class_names[3]}: {best_f1_svm[3]:.5f}, {class_names[4]}: {best_f1_svm[4]:.5f}")
    print(f"Best SVM F1-score: {f1_svm}\n")

    print("Visualise the Confusion Matrix\n")
    confusion_matrix_display(flag=0, classifier=svm_clf, X_test=X_test, y_test=y_test)

    # RF classification
    print('Start RF classification')
    rf_clf, rf_results, rf_best_params, rf_best_acc = RF_classification(
        X_train, X_test, y_train, y_test, chosen_model, chosen_params_rf
    )

    best_f1_rf = f1_score(y_test, rf_clf.predict(X_test), average=None, labels=[0,1,2,3,4])
    f1_rf = f1_score(y_test, rf_clf.predict(X_test), average='macro')

    print(f"Best RF F1-score for each class: {class_names[0]}: {best_f1_rf[0]:.5f}, {class_names[1]}: {best_f1_rf[1]:.5f}, {class_names[2]}: {best_f1_rf[2]:.5f}, {class_names[3]}: {best_f1_rf[3]:.5f}, {class_names[4]}: {best_f1_rf[4]:.5f}")
    print(f"Best RF F1-score: {f1_rf}\n")

    print("Visualize the Confusion Matrix\n")
    confusion_matrix_display(flag=1, classifier=rf_clf, X_test=X_test, y_test=y_test)

    print("Generating learning curve\n")
    learning_curve(X_raw, y, best_features, svm_best_params, rf_best_params)


    print("Summary:")
    print(f"Selected feature names {best_features}:", selected_feature_names)
    print(f"Best SVM hyperparameters:, {svm_best_params}, Best RF hyperparameters:, {rf_best_params}")
    print(f"Best SVM accuracy: {svm_best_acc:.5f}, Best RF accuracy: {rf_best_acc:.5f}")
    print(f"Best SVM f1-score: {f1_svm:.5f}, Best RF f1-score: {f1_rf:.5f}")


