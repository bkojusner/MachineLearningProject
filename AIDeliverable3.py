from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from os.path import dirname, join
import csv
import numpy as np

class Bunch(dict):
    def __init__(self, **kwargs):
        super(Bunch, self).__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass

def load_data(module_path, data_file_name):
    with open(join(module_path, data_file_name)) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)

    return data, target, target_names

def load_winequality(return_X_y=False):
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'winequality-red.csv')
    winequality_csv_filename = join(module_path, 'data', 'winequality-red.csv')

    fdescr = ""

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality'],
                 filename=winequality_csv_filename)

# size of testing set (percentage, 0 -> 1)
size_test = 0.5

# kernel name (SVM has various kernels: linear, poly, rbf)
kernel_name = 'linear'

wine = datasets.load_wine()
iris = datasets.load_iris()
breast_cancer = datasets.load_breast_cancer()
winequality = load_winequality()

for dataset, name in zip((wine, iris, breast_cancer, winequality),('wine','iris','breast cancer','wine quality')):
    # set X to be the features (there are 13)
    X = dataset.data

    # set y to be the target (0 = malignant, 1 = benign)
    y = dataset.target

    # set up the training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size_test) 

    # training
    svclassifier = SVC(kernel = kernel_name, gamma = 'auto')
    svclassifier.fit(X_train, y_train)

    # testing
    y_pred = svclassifier.predict(X_test)

    confusion = confusion_matrix(y_test, y_pred)
    # print to find out accuracy
    print("------- {}, {} --------".format(name, kernel_name))
    print(confusion)
    hit = np.trace(confusion)
    miss = np.sum(confusion) - hit
    percent = (hit/(hit+miss))*100
    print("hits: {}, misses: {}, percentage hit: {}%".format(hit, miss, percent))
    print(classification_report(y_test, y_pred))
