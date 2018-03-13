from __future__ import division

__author__ = "Alexis Coffer"
__copyright__ = "March 2018"
__credits__ = "Bio_393_Compbio"
__version__ = "1.0"
__maintainer__ = "__author__"
__email__ = "acoffer@uw.edu"
__status__ = "Production"
__file__ = "/Users/alexiscoffer/Documents/UW_Winter_2018/project_2_28Feb2018/magnoliopsida.csv"

import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

def make_commandline_interface():
    """Returns a parser for the commandline"""
    short_description = \
        """
        mass_extinctions.py; analyze the impact of earth imapacts and volcanism on mass extinctions

        """

    long_description = \
        """
        Comparitive analysis between earth impact dates and dates of mass extinctions
        in the genus Magnoliopsida. CSV of earth impact dates was used to iterate over
        earth stages dictionary. THis data was then used to calculate a data set
        using linear regression models to perform this analysis of the genus Magnoliopsida.


        Background
        ---------
        Magnoliopsida was used in this analyis due to it's wide spread occurance
        over multiple time periods and geographical locations. Magnoliopsida is also
        a genus that can still be found on earth today.


        References
        ----------
        PaleoDB Extinction Dataset: https://paleobiodb.org/navigator/
        Article for reference: http://geoweb.princeton.edu/research/Paleontology/Keller_AJES_05.pdf
        Volcano Occurance Dataset: https://volcano.si.edu/search_eruption.cfm
        PyRate Tutorial: https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_1.md
        Earth Impact Database: http://www.passc.net/EarthImpactDatabase/zhamanshin.html
        Geological timline Dataset: https://web.archive.org/web/20040623025505/http://www.stratigraphy.org/geowhen/
        Magnoliopsida References: http://botit.botany.wisc.edu/courses/systematics/Phyla/Magnoliophyta/Magnoliopsida.html


        Notes
        ----------
        Data Set Characteristics:
        Records Found for Magnoliopsida 20720
        Records Returned for Magnoliopsida 20720

        """

    parser = ArgumentParser(description=short_description, \
        epilog=long_description)

    parser.add_argument('--version', action='version', version=__version__, \
        help="display version number and exit")

    return parser


def load_magnoliopsida(return_X_y=False):
    """Load and return the diabetes dataset (regression).
    ==============      ==================
    Samples total       20720
    Dimensionality      11
    Features            real, -.2 < x < .2
    Targets             integer 0 - 346
    ==============      ==================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn and 'target', the regression target for each
        sample.
    (data, target) : tuple if ``return_X_y`` is True
        """
    module_path = dirname(__file__)
    base_dir = join(module_path, 'data')
    data = np.loadtxt(join(base_dir, 'magnoliopsida.csv'))
    target = np.loadtxt(join(base_dir, 'magnoliopsida.csv'))

    with open(join(module_path, 'descr', 'magnoliopsida.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target, DESCR=fdescr,
            feature_names=['max_ma', 'min_ma','X_Ft','C_bL', 'X_FL','X_bt','a1','b1','c1','d1','e1','f1'])


#Load the paleoDB datasets
magnoliopsida = pd.read_csv('magnoliopsida.csv')
magnoliopsida_X = magnoliopsida.iloc[:, 6:10].values
magnoliopsida_Y = magnoliopsida.iloc[:, 0:4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(magnoliopsida_X, magnoliopsida_Y, test_size = 0.25)

#create linear regression object
regr = linear_model.LinearRegression()

#training the model using the training sets
regr.fit(X_train, y_train)

#make predicitons using the testing sets
y_train = regr.predict(X_test)

#the coefficients
print('coefficients: \n', regr.coef_)

#the mean squared error
print("Mean squared error: %.2f"
    %mean_squared_error(y_test, y_train))

#explained variance score: 1 is perfect prediction
print('variance score: %.2f' % r2_score(y_test, y_train))

#plot outputs
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_train, color='red', linewidth=3)

plt.xticks(())
plt.yticks(())


plt.show()


if __name__ == "__main__":
    main()
