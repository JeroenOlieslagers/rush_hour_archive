import pandas as pd
import matplotlib
from sklearn import linear_model

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from analyze_mags import MOVE_FILE_WITH_MAG, MOVEDATA_FILE


def lr_on_multiple_factors(df):

    r = linear_model.LogisticRegression().fit(X=df[['optlen', 'rt']], y=df.error_made)
    print(r)


if __name__ == '__main__':
    """
    Example LR on moves_file
    """
    df = pd.read_csv(MOVEDATA_FILE)
    lr_on_multiple_factors(df)

    """
    Reading from the file with stats on mags
    """
    df = pd.read_csv(MOVE_FILE_WITH_MAG)
    sns.distplot(df.number_of_nodes)
    plt.show()

    sns.distplot(df.number_of_leaves)
    plt.show()

