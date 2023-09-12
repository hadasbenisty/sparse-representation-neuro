import torch
import torch.nn.functional as F
import numpy as np                 # import numpy
import matplotlib.pyplot as plt    # import matplotlib
from torch.utils.data import Dataset, DataLoader,random_split
import torch.optim as optim
from tqdm import tqdm
import h5py
import scipy
from scipy.signal import find_peaks
from IPython.display import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import optuna

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(f"using {device} as device")


class load_dataset(Dataset):
    def __init__(self, path,device):
        data = scipy.io.loadmat(path)['imagingData']['samples'][0,0]
        self.y = torch.zeros((len(data), data[0].shape[0]*data[0].shape[1]), device=device)
        for i in range(len(data)):
          self.y[i] = torch.tensor(data[i].flatten())
        self.y = torch.unsqueeze(self.y, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.y[idx]


def load_behave_data(path, metrics = ['lift','grab','atmouth','tone','supination']):
    data = scipy.io.loadmat(path)['BehaveData']
    field_names = data.dtype.names
    behavior_dict = {}
    for metric in metrics:
      behave = None
      for field in field_names:
        if metric in field:
          events = np.concatenate(scipy.io.loadmat(path)['BehaveData'][field][0][0][0][0][0])
          if behave is None:
            behave = np.zeros((len(events),))
          behave += events
      behavior_dict[metric] = behave > 0 #there is event overlap

    return behavior_dict


def f1_score(label, pred):
    label = np.array(label)
    true_positives = np.sum((pred == 1) & (label == 1))
    all_positives = np.sum(label == 1)
    predicted_positives = np.sum(pred == 1)

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    recall = true_positives / all_positives if all_positives > 0 else 0.0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def check_naive_classifiers(signal, label, k=10):
    dataset_behave = create_behave_dataset_matrix(signal,label,device = device)
    X = dataset_behave.data.cpu()
    y = dataset_behave.label

    kf = KFold(n_splits=k, shuffle=False)
    test_accuracy_list = []
    print("running tuna study on metric")
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)

        def objective(trial):
            # Define hyperparameter search space
            model_type = trial.suggest_categorical('model', ['svm_linear', 'svm_rbf', 'svm_poly', 'logistic'])
            C = trial.suggest_float('C', 0.01, 100, log=True)
            degree = trial.suggest_int('degree', 2, 15) if model_type == 'svm_poly' else None

            if model_type.startswith('svm'):
                if model_type == 'svm_linear':
                    svm_classifier = svm.SVC(C=C, kernel='linear')
                elif model_type == 'svm_rbf':
                    svm_classifier = svm.SVC(C=C, kernel='rbf')
                else:
                    svm_classifier = svm.SVC(C=C, kernel='poly', degree=degree)
                svm_classifier.fit(X_train, y_train)
                y_val_pred = svm_classifier.predict(X_val)
                val_accuracy = f1_score(y_val, y_val_pred)
            else:
                logistic_classifier = LogisticRegression(C=C, max_iter=1000)
                logistic_classifier.fit(X_train, y_train)
                y_val_pred = logistic_classifier.predict(X_val)
                val_accuracy = f1_score(y_val, y_val_pred)

            return val_accuracy

        # Create an Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        # Get the best trial
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value

        # Train the best model on the entire training set
        if best_params['model'].startswith('svm'):
            best_model = svm.SVC(C=best_params['C'], kernel=best_params['model'][4:])
        else:
            best_model = LogisticRegression(C=best_params['C'], max_iter=1000)
        best_model.fit(X_train, y_train)

        # Evaluate the model on the test set
        y_test_pred = best_model.predict(X_test)
        test_accuracy = f1_score(y_test, y_test_pred)

        print("Best Hyperparameters from Optuna:", best_params)
        print("Best Validation Set Accuracy:", best_score)
        print("Test Set Accuracy:", test_accuracy)
        test_accuracy_list.append(test_accuracy)

    return np.mean(test_accuracy_list)


class create_behave_dataset_matrix(Dataset):
    def __init__(self, x, label, device):
        self.label = label
        print("normalizing data")
        row_means = torch.mean(x, dim=1, keepdim=True)
        row_variances = torch.var(x, dim=1, keepdim=True)
        # Subtract mean and divide by variance for each row
        normalized_x = torch.where(row_variances != 0, (x - row_means)/row_variances, 0)
        # Concatenating the chunks along the first dimension
        self.data = normalized_x.T

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return  self.data[idx], self.label[idx]


class CRsAE1D(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(CRsAE1D, self).__init__()

        self.T = hyp["T"]            # number of encoder unfolding
        self.L = hyp["L"]            # 1/L is the step size in the encoder
        self.C = hyp["C"]            # number of filters
        self.K = hyp["K"]            # legnth of the filters
        self.lam = hyp["lam"]        # lambda (regularization parameter)
        self.device = hyp["device"]  # device (i.e., cpu, cuda0)

        # initialize the filter H
        if H is None:
            # initialize with random normal
            H = torch.randn((self.C, 1, self.K), device=self.device)
            # normalize that each filter has norm 1
            H = F.normalize(H, p=2, dim=-1)
        # register the filters as weights of
        # the neural network so that to be trainable.
        self.register_parameter("H", torch.nn.Parameter(H))
        # create ReLU
        self.relu = torch.nn.ReLU()

    def get_param(self, name):
        # get parameters with name
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        # normalize the filters
        self.get_param("H").data = F.normalize(self.get_param("H").data, p=2, dim=-1)

    def H_operator(self, x):
        return F.conv_transpose1d(x, self.get_param("H"))

    def HT_operator(self, x):
        return F.conv1d(x, self.get_param("H"))

    def encoder(self, y):
        enc_dim = F.conv1d(y, self.get_param("H")).shape[-1]

        x_old = torch.zeros(y.shape[0], self.C, enc_dim, device=self.device)
        x_tmp = torch.zeros(y.shape[0], self.C, enc_dim, device=self.device)
        x_new = torch.zeros(y.shape[0], self.C, enc_dim, device=self.device)
        s_old = torch.tensor(1, device=self.device).float()

        # T recurrent steps
        for t in range(self.T):
            res = y - self.H_operator(x_tmp)

            x_new = x_tmp + self.HT_operator(res) / self.L

            x_new = self.relu(x_new - self.lam / self.L)

            s_new = (1 + torch.sqrt(1 + 4 * s_old * s_old)) / 2
            x_tmp = x_new + (s_old - 1) / s_new * (x_new - x_old)

            x_old = x_new
            s_old = s_new
        return x_new

    def decoder(self, x):
        return F.conv_transpose1d(x, self.get_param("H"))

    def forward(self, y):
        # encoder
        x = self.encoder(y)
        # decoder
        y_hat = self.decoder(x)
        return y_hat, x

    def separate(self,y):
        with torch.no_grad():
            # encoder
            x = self.encoder(y)

            hx_separate = torch.zeros((y.shape[0], self.C, y.shape[-1]), device=self.device)
            for c in range(self.C):
                xc = torch.unsqueeze(x[:,c,:], dim=1)
                hc = torch.unsqueeze(self.get_param("H")[c,:,:],dim=0)
                hx_separate[:,c,:] =  torch.squeeze(F.conv_transpose1d(xc, hc),dim=1)

        return hx_separate


if __name__ == '__main__':

    net_hyp = {"T": 800,                 # number of encoder unfolding
               "L": 10,                  # 1/L is the step size in the encoder
               "C": 1,                 # number of filters
               "K": 5,                  # legnth of the filters
               "lam": 1.5,
               "device": device
               }

    net = CRsAE1D(net_hyp)
    print("Loading pretrained network")
    # Load the saved state dictionary
    net.load_state_dict(torch.load('real_data_pretrained_network.pth')) # if run on GPU

    # net.load_state_dict(torch.load('real_data_pretrained_network.pth',map_location=torch.device('cpu')))  # if run on cpu

    print("Creating dataset")
    with torch.no_grad():
        dataset = load_dataset(r'./data.mat',device = device)
        y = dataset[:]
        print("putting data trough network")
        y_hat, x_hat = net(y)
        y = y.squeeze(dim = 1)
        y_hat = y_hat.squeeze(dim = 1)
        x_hat = x_hat.squeeze(dim = 1)

    y_accuracy = []
    y_hat_accuracy = []

    metrics = ['grab']
    print("Creating behave dataset")
    dataset_behave = load_behave_data(r'./data.mat',metrics = metrics)
    for metric in metrics:
        print(f"calculating F1 score on {metric} action")
        #print("%%%%% On y %%%%%")
        #y_accuracy.append(check_naive_classifiers(y, dataset_behave[metric]))
        #print(y_accuracy)
        print("%%%%% On y_hat %%%%%")
        y_hat_accuracy.append(check_naive_classifiers(y_hat, dataset_behave[metric]))
        print(y_hat_accuracy)
        #print("%%%%% On x_hat %%%%%")
        #x_hat_accuracy = check_naive_classifiers(x_hat, dataset_behave[metric])