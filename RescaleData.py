####### Źródło:   https://metsi.github.io/2020/05/15/kod8.html   ###################

import numpy
import pandas
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour
from sklearn import clone
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from math import pi

####### Źródło:   https://metsi.github.io/2020/05/15/kod8.html   ###################

url = "D:/StudiaMagSem2/uczenieMaszynProjekt/glass1.dat"
dataframe = pandas.read_csv(url)
array = dataframe.values
tmp = array[:, -1]
tmp2 = tmp == 'positive'

X = array[:, :-1]
y = tmp2.astype(int)

# scaler = MinMaxScaler(feature_range=(0, 1))
# rescaledX = scaler.fit_transform(X)
# numpy.set_printoptions(precision=3)
# X = rescaledX
print(X[0:5, :])


clf = MLPClassifier(random_state=2323, max_iter=10000)

preprocs = {
    'none': None,
    'ros': RandomOverSampler(random_state=2323),
    'smote': SMOTE(random_state=2323),
    'rus': RandomUnderSampler(random_state=2323),
    'cnn': CondensedNearestNeighbour(random_state=2323),
}
metrics = {
    "recall": recall,
    'precision': precision,
    'specificity': specificity,
    'f1': f1_score,
    'g-mean': geometric_mean_score_1,
    'bac': balanced_accuracy_score,
}

n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=2323)

scores = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for preproc_id, preproc in enumerate(preprocs):
        clf = clone(clf)

        if preprocs[preproc] == None:
            X_train, y_train = X[train], y[train]
        else:
            X_train, y_train = preprocs[preproc].fit_resample(
                X[train], y[train])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X[test])

        for metric_id, metric in enumerate(metrics):
            scores[preproc_id, fold_id, metric_id] = metrics[metric](
                y[test], y_pred)

# Zapisanie wynikow

np.save('results', scores)

scores = np.load("results.npy")

scores = np.mean(scores, axis=1).T




metrics=["Recall", 'Precision', 'Specificity', 'F1', 'G-mean', 'BAC']
methods=["None", 'ROS', 'SMOTE', 'RUS', 'CNN']
N = scores.shape[0]


angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]


ax = plt.subplot(111, polar=True)


ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)


plt.xticks(angles[:-1], metrics)

# os y
ax.set_rlabel_position(0)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
color="grey", size=7)
plt.ylim(0,1)


for method_id, method in enumerate(methods):
    values=scores[:, method_id].tolist()
    values += values[:1]
    print(values)
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=method)


plt.legend(bbox_to_anchor=(1.15, -0.05), ncol=5)

plt.savefig("radar", dpi=200)


# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=3333)
#
# clf.fit(x_train, y_train)
# print(clf.score(x_test, y_test))
#
# accuracy1 = accuracy_score(y_test, clf.predict(x_test))
# print(accuracy1)