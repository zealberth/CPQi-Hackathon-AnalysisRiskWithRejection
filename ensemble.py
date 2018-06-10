import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from scipy import stats


class Ensemble:
    '''
        Ensemble model is a machine learning where a set of classifiers are trained
        to solve the same problem by taking a vote (weighted or not) of each
        classifier.
    '''

    def train(self, input_train, output_train, gridSearch=False):
        """
        Train the following set of classifiers: MLP, KNN, Naive Bayes, SVM and
        Gaussian Mixture Models. We have set default values based on Grid Search
        with k-fold cross-validation with k = 5

        :param input_train: training data
        :param output_train: output of training data
        :return: Ensemble model trained
        """

        # Classifiers applied to gridSearch
        if gridSearch:
            # MLP classifier
            params_MLP = [
                {'hidden_layer_sizes': [20, 30, 50, 70, 100, 130, 150],
                 'activation': ['tanh', 'relu']}]
            mlp = MLPClassifier(solver='lbfgs', random_state=1,
                                activation='tanh')
            self.mlp = GridSearchCV(estimator=mlp, param_grid=params_MLP,
                                    scoring='accuracy', cv=5)

            #  KNN classifier
            params_KNN = [
                {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}]
            knn = KNeighborsClassifier()
            self.knn = GridSearchCV(estimator=knn, param_grid=params_KNN,
                                    scoring='accuracy', cv=5)

            #  Naive Bayes classifier
            self.nb = GaussianNB()

            #  Gaussian Mixture Models (GMM) classifier
            params_GMM = [{'n_components': [1, 2, 3, 5, 7],
                           'covariance_type': ['full', 'tied']}]

            gmm = GaussianMixture()
            self.gmm = GridSearchCV(estimator=gmm, param_grid=params_GMM,
                                    scoring='accuracy', cv=5)

            # SVM classifier
            interval = np.float_power(2, np.arange(-5, 10))
            params_SVM = [
                {'C': interval, 'gamma': interval, 'kernel': ['rbf']}]
            self.svm = GridSearchCV(SVC(max_iter=1000), params_SVM, cv=5,
                               scoring='accuracy')

        # Classifiers with default values
        else:
            #  MLP classifier
            self.mlp = MLPClassifier(solver='lbfgs', random_state=1,
                                     activation='tanh', hidden_layer_sizes=30,
                                     alpha=1e-05)

            #  KNN classifier
            self.knn = KNeighborsClassifier(n_neighbors=15)

            #  Naive Bayes classifier
            self.nb = GaussianNB()

            #  Gaussian Mixture Models (GMM) classifier
            self.gmm = GaussianMixture(covariance_type='full', n_components=1)

            # SVM classifier
            self.svm = SVC(C=0.25, gamma=4.0, kernel='rbf')

        # training models
        self.mlp.fit(input_train, output_train)
        self.knn.fit(input_train, output_train)
        self.nb.fit(input_train, output_train)
        self.gmm.fit(input_train, output_train)
        self.svm.fit(input_train, output_train)


    def predict(self, input_test):
        '''

        :param input_test: data to be tested
        :return: predicted values
        '''

        result_mlp = self.mlp.predict(input_test)
        result_knn = self.knn.predict(input_test)
        result_nb = self.nb.predict(input_test)
        result_svm = self.svm.predict(input_test)
        result_gmm = self.gmm.predict(input_test)

        result = np.stack((result_mlp, result_knn, result_nb, result_gmm,
                           result_svm))

        y_hat_size = np.shape(result)[1]
        y_hat = np.zeros((y_hat_size,), dtype=int)

        for col_predictions in range(y_hat_size):
            # Instead of calculating Gini Indices, we accept a threshold value
            # to define whether or not our Ensemble model reject or classify
            # a pattern.

            threshold = sum(result[:, col_predictions])
            if threshold == 2 or threshold == 3:
                y_hat[col_predictions] = 2
            else:
                y_hat[col_predictions] = stats.mode(result[:, col_predictions])[0]
        return y_hat


    def evaluate(self, y_hat, y_test):

        rejection_rate = len(y_hat[np.where(y_hat == 2)[0]]) / len(y_hat)
        error_rate = np.mean(
            y_test[np.where(y_hat != 2)[0]] != y_hat[np.where(y_hat != 2)[0]])

        return error_rate, rejection_rate
