import tensorflow as tf
from tensorflow.contrib.slim import fully_connected
from tensorflow import losses, nn
import numpy as np
import os

# MODEL_HYPER_PARAM
N_FC1 = 100
N_FC2 = 200
N_FC3 = 100
N_FC4 = 20
N_OUTPUT = 1
# LOSS_FUNC = nn.l2_loss
LOSS_FUNC = losses.mean_squared_error

# COLUMN = ['airconditioningtypeid',
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


class property_price_regression:
    def __init__(self, learning_rate, input_size):
        self.input_size = input_size
        self.session = tf.Session()
        self.x = tf.placeholder(tf.float32, (None, input_size))
        self.y = tf.placeholder(tf.float32, shape=(None, 1))
        self.fc1 = fully_connected(inputs=self.x, num_outputs=N_FC1)
        self.fc2 = fully_connected(inputs=self.fc1, num_outputs=N_FC2)
        self.fc3 = fully_connected(inputs=self.fc2, num_outputs=N_FC3)
        self.fc4 = fully_connected(inputs=self.fc3, num_outputs=N_FC4)
        self.output = fully_connected(
            inputs=self.fc4, num_outputs=N_OUTPUT, activation_fn=None)
        self.loss = self.get_loss()
        self.minimize_loss = self.minimizor(self.loss, learning_rate)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def minimizor(self, loss, learning_rate):
        return tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer") \
            .minimize(loss)

    def get_loss(self):
        # loss = LOSS_FUNC(labels=self.y, predictions=self.output)
        loss = LOSS_FUNC(self.y, self.output)
        # loss = tf.Print(loss, [loss])
        return loss

    def train(self, dataset, batch_size, epoch):
        data_placeholder = tf.placeholder(tf.float32, (None, self.input_size))
        label_placeholder = tf.placeholder(tf.float32, (None, 1))

        formal_dataset = tf.data.Dataset.from_tensor_slices((data_placeholder, label_placeholder))
        iter = formal_dataset.batch(batch_size) \
            .shuffle(10000) \
            .make_initializable_iterator()
        batch_x, batch_y = iter.get_next()
        step = 0
        for e in range(epoch):
            self.session.run(iter.initializer, feed_dict={
                data_placeholder: dataset[:, :-1],
                label_placeholder: dataset[:, -1][:, np.newaxis]
            })
            # re-init iterator and
            while True:
                try:
                    d, l = self.session.run([batch_x, batch_y])
                    # take a batch from dataset
                    step += 1
                    l, _ = self.session.run([self.loss, self.minimize_loss], feed_dict={
                        self.x: d,
                        self.y: l
                    })
                    if step % 1000:
                        step = 0
                        print('epoch:', e, 'loss:', l)
                    # send it to train network
                except tf.errors.OutOfRangeError:
                    break

    def predict(self, input):
        return self.session.run(self.output, feed_dict={
            self.x: input
        })

    def dump(self):
        new_dir = os.path.join(os.getcwd(), 'model')
        os.mkdir(new_dir)
        self.saver.save(self.session, os.path.join(new_dir, 'trained_model.pb'))


if __name__ == '__main__':
    network = property_price_regression(23, 0.001)
    network.dump()
