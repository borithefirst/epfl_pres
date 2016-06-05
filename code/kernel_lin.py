import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.svm import SVC
from sklearn.kernel_approximation import Nystroem, RBFSampler
import math
import gflags
import sys

FLAGS = gflags.FLAGS

gflags.DEFINE_bool(
    'kernel_svc',
    False,
    'Use a kernel SVC as learning method (in addition to feature transforms...)',
)

gflags.DEFINE_string(
    'model_params',
    '{}',
    'Model parameters dictionary...',
)

gflags.DEFINE_bool(
    'rbfsampler',
    False,
    'Uses a kernel transform based on RBF fourrier approx',
)

gflags.DEFINE_bool(
    'boosting',
    False,
    'Uses a boosting transform',
)
gflags.DEFINE_bool(
    'nystroem',
    False,
    'Uses a nystroem kernel transform (specify metric with --metric)',
)

gflags.DEFINE_bool(
    'augment',
    False,
    'Uses a feature augmentation (keep original features)',
)

gflags.DEFINE_bool(
    'erase_base',
    False,
    'Erase original feature space information',
)

gflags.DEFINE_bool(
    'plot_labels',
    False,
    'Plot labels (scatter only...)',
)

gflags.DEFINE_bool(
    'scatter',
    False,
    'Plot a scatter plot',
)

gflags.DEFINE_bool(
    'heatmap',
    False,
    'Plot a heatmap plot',
)

gflags.DEFINE_integer(
    'N',
    300,
    'Number of samples to use',
)

gflags.DEFINE_string(
    'metric',
    'rbf',
    'Kernel metric',
)

gflags.RegisterValidator(
    'metric',
    lambda x: x in ['rbf', 'sigmoid', 'polynomial', 'poly', 'linear', 'cosine'],
    "metric must be in ['rbf', 'sigmoid', 'polynomial', 'poly', 'linear', 'cosine']"
)

gflags.DEFINE_float(
    'xmax',
    16.0,
    'Heatmap plot limit',
)
gflags.DEFINE_float(
    'gamma',
    0.2,
    'Kernel size parameter, as in exp(-gamma d^2) for RBF',
)
gflags.DEFINE_string(
    'dataset',
    'arcs',
    'Dataset: circles, or arcs',
)

gflags.DEFINE_integer(
    'irrelevant_dimensions',
    0,
    'Adds random, irrelevant dimensions to training and prediction.'
)
gflags.DEFINE_float(
    'irrelevant_dimensions_variance',
    5.0,
    'variance of random additional dimensions'
)

gflags.DEFINE_integer(
    'kernel_sampling',
    -1,
    'Activates kernel sampling up to this many dimensions.',
)

gflags.DEFINE_float(
    'mixing',
    0.0,
    'How much do the classes mix, 0 to 1.0'
)

def circle(r, n, angle_limit=(0.0, 2 * np.pi), bias=(0.0, 0.0)):
    angles = np.random.uniform(angle_limit[0], angle_limit[1], n)
    radii = np.random.normal(r, 1.0, n)

    return np.array(
        (
            radii * np.cos(angles) + bias[0],
            radii * np.sin(angles) + bias[1],
        ),
    ).T


def get_Xy(N):
    if FLAGS.dataset == 'circles':
        c1 = circle(8.0, N)
        c2 = circle(16.0, N)
    elif FLAGS.dataset == 'arcs':
        r = 1 - FLAGS.mixing
        c1 = circle(r * 20.0, N, angle_limit=(-np.pi / 2, np.pi / 2), bias=(-5 * r, 10.0 * r))
        c2 = circle(r * 20.0, N, angle_limit=(np.pi / 2, 3 * np.pi / 2), bias=(5 * r, -10.0 * r))
    else:
        raise ValueError("Unknown dataset " + FLAGS.data)

    return (
        pd.DataFrame(np.concatenate((c1, c2), axis=0), columns=['x', 'y']),
        np.array([0.0] * N + [1.0] * N)
    )


def add_irrelevant_dimensions(X):
    if FLAGS.irrelevant_dimensions <= 0:
        return X

    return np.concatenate(
        (
            X,
            np.random.normal(
                0.0,
                FLAGS.irrelevant_dimensions_variance,
                size=(
                    X.shape[0],
                    FLAGS.irrelevant_dimensions,
                ),
            ),
        ),
        axis=1,
    )

def plot(plotsample, score=0.0):
    if FLAGS.heatmap:
        ax = sns.heatmap(plotsample.pivot('y', 'x', 'c'), xticklabels=32, yticklabels=32)
        ax.invert_yaxis()

    elif FLAGS.scatter:
        ax = plotsample.plot.scatter('x', 'y', c='c', s=50)
        if score > 0:
            ax.set_title('Score {:.03f}'.format(score))

    pyplot.show()

class GRDTransformer:
    def __init__(self, n_estimators):
        self.model = GradientBoostingClassifier(n_estimators=n_estimators)
        self.enc = OneHotEncoder(sparse=False)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.enc.fit(self.model.apply(X)[:, :, 0])

    def transform(self, X):
        return self.enc.transform(self.model.apply(X)[:, :, 0])

def apply_trf(trfs, X):
    if not trfs:
        return X

    transformed = [
        trf.transform(X)
        for trf in trfs
    ]
    if FLAGS.augment:
        transformed.append(X)
    return np.concatenate(transformed, axis=1)

def get_model():
    params = eval(FLAGS.model_params)

    if FLAGS.kernel_svc:
        params['probability'] = True
        return SVC(**params)

    return LogisticRegression(**params)

def heatmap_grid():
    ppts = np.around(
        np.linspace(-FLAGS.xmax, FLAGS.xmax, 128),
        decimals=2,
    )
    return pd.DataFrame([
            (x, z)
            for x in ppts
            for z in ppts
        ],
        columns=['x', 'y'],
    )


def main(argv):
    FLAGS(argv)
    N = FLAGS.N
    X, y = get_Xy(FLAGS.N)

    # Kernel features to build.
    kd = (
        2 * N if FLAGS.kernel_sampling == -1
        else FLAGS.kernel_sampling
    )

    trfs = []
    # Feature engineering random kitchen sinks, boosting or Nystroem (default)
    if FLAGS.rbfsampler:
        trfs.append(RBFSampler(n_components=kd, gamma=FLAGS.gamma))
    if FLAGS.boosting:
        trfs.append(GRDTransformer(n_estimators=5))
    if FLAGS.nystroem:
        trfs.append(Nystroem(kernel=FLAGS.metric, n_components=kd, gamma=FLAGS.gamma))

    X_train = X
    if FLAGS.irrelevant_dimensions > 0:
        X_train = add_irrelevant_dimensions(X)

    # Fit transforms
    for trf in trfs:
        trf.fit(X_train, y)

    X_train = apply_trf(trfs, X_train)

    m = get_model()

    m.fit(X_train, y)
    if FLAGS.erase_base:
        m.coef_[0,-(2 + FLAGS.irrelevant_dimensions)] = 0.0
        m.coef_[0,-(1 + FLAGS.irrelevant_dimensions)] = 0.0

    if FLAGS.heatmap:
        plotsample = heatmap_grid()
        y_test = np.zeros(len(plotsample))
    else:
        plotsample, y_test = get_Xy(FLAGS.N)


    # Adds Irrelevant dimensions to prediction set, or nothing if there are none
    X_test = add_irrelevant_dimensions(plotsample)
    # Adds feature transforms.
    X_test = apply_trf(trfs, X_test)

    score = -1.0
    if FLAGS.plot_labels:
        plotsample['c'] = y_test

    else:
        score = m.score(X_test, y_test)
        plotsample['c'] = pd.Series(
            m.predict_proba(X_test)[:,0]
        )

    plot(plotsample, score)

if __name__ == '__main__':
    main(sys.argv)
