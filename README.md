# CPQi-Hackathon-AnalysisRiskWithRejection

## TEAM: Atilla Maia, Francisco Felipe, Jose Alberth

Our problem is related to risk analysis, where we have to classify whether or not a customer will pay a loan based on

past payments and personal attributes (profile).

The dataset used in this hackathon is available in UCI Machine Learning (https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

About the dataset, it is unbalanced, approximately 22% of the samples belongs to one class (0 - non-risky). Therefore, another dataset was generated

(through the reduction of the class with more amount of patterns), in order to balance (generalize) the data. Experiments were performed using both datasets, 20 executions for each dataset .

The classification model is based on Ensemble learning. The model consists of the following methods: Support Vector Machine (SVM), MultiLayer Perceptron (MLP), Naive Bayes (NB), Gaussian Mixture Models (GMM),

and k-Nearest Neighborhood (KNN) classifiers. The vote of each classifier has weight 1. Also, we developed a Reject Option method with the Ensemble learning.

Thus, the ensemble model will only classify a sample when at least four classifiers agree (at least 80% of classifiers). When the ensemble diverges (3 or 2 against), the sample is rejected, which means it is

selected to be classified later by an expert in the area.

The training phase of each model is applied Grid Search with k-fold cross-validation (k = 5) in order to find the best parameters.

The script 'run.py' execute the Ensemble model.