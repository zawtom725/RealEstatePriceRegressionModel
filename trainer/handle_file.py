from google.cloud import storage
import os
import numpy as np

client = storage.Client()


def get_dataset(bucket_name, dir_path, new_dir_name):
    os.mkdir(new_dir_name)
    allfiles = client.bucket(bucket_name).list_blobs(prefix=dir_path)
    for step, i in enumerate(allfiles):
        if step != 0:
            i.download_to_filename(new_dir_name + '/' + str(step) + '.csv')


def save_data(bucket_name, filename):
    b = client.bucket(bucket_name)
    m = b.blob(filename)
    m.upload_from_filename(filename)


if __name__ == '__main__':
    dataset = get_dataset('property_nn', 'train', 'train')
    training_files = ['train' + '/' + i for i in os.listdir('train')]
    dataset = np.concatenate([np.genfromtxt(i, delimiter=',') for i in training_files], axis=0)
