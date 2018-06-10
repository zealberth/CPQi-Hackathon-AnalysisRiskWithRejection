import os
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt

from ensemble import Ensemble
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def main():
    # Dataset path
    dataset_name = ['credit_card_clients_balanced',
                    'credit_card_clients']

    for data_name in dataset_name:
        dataset_path = os.getcwd() + "\\dataset\\" + data_name + ".csv"
        dataset = pd.read_csv(dataset_path, encoding='utf-8')

        # Datasets columns
        data_x = dataset[
            ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11',
             'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21',
             'X22', 'X23']]
        data_y = dataset['Y']

        # Preprocessing data
        min_max_scaler = preprocessing.MinMaxScaler()
        X_normalized = min_max_scaler.fit_transform(data_x)

        acc_rate = []
        reject_rate = []

        # Runs to test the model
        for i in range(20):
            print('---------------- Ensemble -----------------')
            print('--- MLP - SVM - KNN - GMM - Naive Bayes ---')
            print(i + 1, 'of 20 iterations')
            X_train, X_test, y_train, y_test = train_test_split(X_normalized, data_y,
                                                            test_size=0.2)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            model = Ensemble()
            model.train(X_train, y_train, gridSearch=False)
            y_hat = model.predict(X_test)

            error, reject = model.evaluate(y_hat, y_test)
            acc_rate.append(1 - error)
            reject_rate.append(reject)

        graphics(acc_rate, reject_rate, data_name)

def graphics(acc_rate, reject_rate, data_name):
    plt.plot(reject_rate, acc_rate, '*-')
    plt.ylabel('Accuracy rate')
    plt.xlabel('Rejection rate')
    plt.suptitle('Accuracy x Rejection for each iteration')
    plt.savefig(os.getcwd() + '\\graphics\\AccuracyVsRejection_' + data_name +
                ".png")


if __name__ == '__main__':
    main()