import tensorflow as tf
from tensorflow.contrib.slim import fully_connected
import pandas as pd
import sklearn.preprocessing as skp
import numpy as np
from enum import Enum
import model1

DATA = './data'
TRAIN_X = DATA + '/properties_2016.csv'
TRAIN_Y = DATA + '/train_2016_v2.csv'
TEST_X = DATA + '/properties_2017.csv'
TEST_Y = DATA + '/train_2017.csv'
SHORT_X = 'g_g.csv'


class Feature_type(Enum):
    CONT = 1
    DISC = 2


class Feature:
    def __init__(self, name, data, t, num_value=1):
        self.name = name
        self.data = data
        self.type = t
        self.num_value = num_value

    @staticmethod
    def convert_from_list(dataset, feature):
        all_feature = []
        all_data = dataset[[i[0] for i in feature]].dropna()
        # simply drop all the NaN
        for i in feature:
            d = all_data[i[0]].values[:, np.newaxis].astype(np.float32)
            if i[1] == Feature_type.CONT:
                # the tuple should be (name, type)
                all_feature.append(d)
            else:
                # the tuple must be (name, type, max)
                encoder = skp.OneHotEncoder(n_values=i[2])
                all_feature.append(encoder.fit_transform(d).toarray())
        return np.concatenate(all_feature, axis=1)


SELECTED = [('airconditioningtypeid', Feature_type.DISC, 13),
            ('bathroomcnt', Feature_type.CONT),
            ('bedroomcnt', Feature_type.CONT),
            ('buildingqualitytypeid', Feature_type.CONT),
            ('calculatedbathnbr', Feature_type.CONT),
            ('calculatedfinishedsquarefeet', Feature_type.CONT),
            ('fips', Feature_type.CONT),
            ('regionidzip', Feature_type.CONT),
            ('roomcnt', Feature_type.CONT),
            ('yearbuilt', Feature_type.CONT),
            ('taxamount', Feature_type.CONT)]  # pick the feature according to feature_selection.xlsx
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('EPOCH', 100, 'how many rounds the model will run')
flags.DEFINE_integer('BATCH_SIZE', 50, 'how many samples used per update')

if __name__ == '__main__':
    df = pd.read_csv(SHORT_X)
    # normalize the dataset:
    # assume all the air conditioning is None if not given, and normalize to
    # 0-index
    df['airconditioningtypeid'] = df['airconditioningtypeid'].fillna(5) - 1
    data = Feature.convert_from_list(df, SELECTED)
