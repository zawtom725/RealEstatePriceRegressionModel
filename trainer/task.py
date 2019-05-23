import argparse
import re
import os
import numpy as np
from sklearn import preprocessing
from trainer import model1, handle_file

CSV = re.compile(r".*csv$")
# COLUMN = ['airconditioningtypeid',
#           'fips',
#           'bathroomcnt',
#           'bedroomcnt',
#           'buildingqualitytypeid',
#           'calculatedbathnbr',
#           'calculatedfinishedsquarefeet',
#           'regionidzip',
#           'roomcnt',
#           'yearbuilt',
#           'taxamount',
#           'taxvaluedollarcnt']

COLUMN = ['airconditioningtypeid',
          'fips',
          'bathroomcnt',
          'lotsizesquarefeet',
          'bedroomcnt',
          'buildingqualitytypeid',
          'calculatedbathnbr',
          'calculatedfinishedsquarefeet',
          'regionidzip',
          'roomcnt',
          'yearbuilt',
          'taxamount',
          'taxvaluedollarcnt']

TRAINING = 'train'
TEST = 'test'
EXPAND = ['airconditioningtypeid', 'fips']
DEPTH = [13, -1]
name_to_col = {name: ind for ind, name in enumerate(COLUMN)}


def encode_one_hot(inputs, to_expand, depth):
    ret = []
    cols = [name_to_col[name] for name in to_expand]
    num_of_col_ascd = sorted(list(zip(cols, depth)), key=lambda x: x[0])
    old_ptr = 0
    new_ptr = 0
    for step, (col, dep) in enumerate(num_of_col_ascd):
        distance = col - old_ptr
        ret.append(inputs[:, old_ptr:old_ptr + distance])
        enc = preprocessing.OneHotEncoder(n_values=dep if dep != -1 else 'auto')
        onehot = enc.fit_transform(inputs[:, col][:, np.newaxis]).toarray()
        old_ptr += 1
        ret.append(onehot)
        new_ptr += onehot.shape[1]
    ret.append(inputs[:, old_ptr:])
    return np.concatenate(ret, axis=1)


def create_np_dataset(files):
    dataset = np.concatenate([np.genfromtxt(i, delimiter=',') for i in files], axis=0)
    return encode_one_hot(dataset, EXPAND, DEPTH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, dest='epoch')
    parser.add_argument('--learning-rate', type=float, dest='lr')
    parser.add_argument('--batch-size', type=int, dest='batch_size')
    parser.add_argument('--job-dir', type=str, dest='job_dir')
    # parser.add_argument('--train-dir', type=str, dest='train')
    # parser.add_argument('--test-dir', type=str, dest='test')

    args = parser.parse_args()
    # expand one-hot vector
    handle_file.get_dataset('property_nn', TRAINING, TRAINING)
    handle_file.get_dataset('property_nn', TEST, TEST)
    training_files = [TRAINING + '/' + i for i in os.listdir(TRAINING) if
                      CSV.match(i) is not None]
    test_files = [TEST + '/' + i for i in os.listdir(TEST) if CSV.match(i) is not None]
    dataset = np.concatenate([np.genfromtxt(i, delimiter=',') for i in training_files], axis=0)
    training_dataset = create_np_dataset(training_files)
    print(training_dataset)
    regression_model = model1.property_price_regression(args.lr, training_dataset.shape[1] - 1)
    regression_model.train(training_dataset, args.batch_size, args.epoch)
    print('training is finished!')
    print('begin prediction')

    # give predict value in 2017
    test_dataset = create_np_dataset(test_files)
    prediction = regression_model.predict(test_dataset[:, :-1])
    result = np.concatenate([prediction, test_dataset[:, -1][:, np.newaxis]], axis=1)
    np.savetxt("result.csv", result, delimiter=',')
    handle_file.save_data('property_nn', 'result.csv')
    # try:
    #     regression_model.dump()
    #     handle_file.save_data('property_nn', './model/trained_model.pb')
    # except Exception as e:
    #     print(e)
