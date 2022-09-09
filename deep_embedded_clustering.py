from time import time
import numpy as np
import pandas as pd
import csv
import keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras import callbacks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def autoencoder(dims, act='relu', init='glorot_uniform'):
    n_stacks = len(dims) - 1
    # INPUT LAYER
    x = Input(shape=(dims[0],), name='input')
    h = x

    # ENCODER LAYERS
    h = Dense(dims[1], activation=act, kernel_initializer=init, name='encoder_0')(h)
    h = Dense(dims[2], activation=act, kernel_initializer=init, name='encoder_1')(h)
    h = Dense(dims[3], activation=act, kernel_initializer=init, name='encoder_2')(h)

    # HIDDEN BOTTLENECK LAYER
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)
    y = h

    # DECODER LAYERS
    y = Dense(dims[3], activation=act, kernel_initializer=init, name='decoder_3')(y)
    y = Dense(dims[2], activation=act, kernel_initializer=init, name='decoder_2')(y)
    y = Dense(dims[1], activation=act, kernel_initializer=init, name='decoder_1')(y)

    # OUTPUT LAYER
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')


class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        # STUDENT'S T DISTRIBUTION
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DEC(object):
    def __init__(self, dims, n_clusters=3, alpha=1.0, init='glorot_uniform'):

        super(DEC, self).__init__()
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder, self.encoder = autoencoder(self.dims, init=init)

        # DEC MODEL
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)

    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256, save_dir='results/temp'):
        print('Pretraining phase started.....')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        if x is not None:
            class PrintACC(callbacks.Callback):
                def __init__(self, x,  cluster_count):
                    self.x = x
                    self.cluster_count = cluster_count
                    super(PrintACC, self).__init__()

                def on_epoch_end(self, epoch, logs=None):
                    if int(epochs/10) != 0 and epoch % int(epochs/10) != 0:
                        return
                    feature_model = Model(self.model.input,
                                          self.model.get_layer(
                                              'encoder_%d' % (int(len(self.model.layers) / 2) - 1)).output)
                    features = feature_model.predict(self.x)
                    km = KMeans(init="k-means++", n_clusters=self.cluster_count)
                    y_pred = km.fit_predict(features)
                    sh_score = silhouette_score(features, y_pred)
                    ch_score = calinski_harabasz_score(features, y_pred)
                    db_score = davies_bouldin_score(features, y_pred)
                    print(' '*8 + '|==>  sh_score: %.4f,  ch_score: %.4f,  db_score: %.4f  <==|'
                          % (sh_score, ch_score, db_score))
                    print("Cluster count: ", pd.Series(y_pred).unique().size)
            cb.append(PrintACC(x, self.n_clusters))

        # START PRETRAINING
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb)
        print('Pretraining time: %ds' % round(time() - t0))
        self.autoencoder.save_weights(save_dir + '/weights.h5')
        print('Pretrained weights are saved to %s/weights.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights):  # load weights of DEC model
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)
        print("DEC Model compilation is complete")

    def fit(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-3, update_interval=100, save_dir='./results/temp'):
        save_interval = int(x.shape[0] / batch_size) * 5
        print(f'Update interval: {update_interval}    Save interval: {save_interval}')

        # STEP 1: INITIALISE CLUSTER CENTERS USING K-MEANS
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(init="k-means++", n_clusters=self.n_clusters)

        enc_output = self.encoder.predict(x)
        y_pred = kmeans.fit_predict(enc_output)
        y_pred_last = np.copy(y_pred)

        self.model.summary()
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # STEP 2: DEEP CLUSTERING
        print('Initializing cluster centers with k-means.')
        logfile = open(save_dir + '/dec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'sh_score', 'ch_score', 'db_score', 'loss'])
        logwriter.writeheader()

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        sh_score = 0.0
        ch_score = 0.0
        db_score = 0.0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)

                # EVALUATION OF CLUSTERING METRICS
                y_pred = q.argmax(1)
                if y_pred is not None and ite > 0:
                    print('y_pred:', y_pred)
                    sh_score = np.round(silhouette_score(enc_output, y_pred), 5)
                    ch_score = np.round(calinski_harabasz_score(enc_output, y_pred), 5)
                    db_score = np.round(davies_bouldin_score(enc_output, y_pred), 5)

                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, sh_score=sh_score, ch_score=ch_score, db_score=db_score, loss=loss)
                    logwriter.writerow(logdict)
                    print('Iter %d: sh_score = %.5f, ch_score = %.5f, db_score = %.5f' %
                          (ite, sh_score, ch_score, db_score), ' ; loss=', loss)
                    print("Cluster count: ", pd.Series(y_pred).unique().size)

                # CHECK STOP CRITERION BASED ON CONVERGENCE THRESHOLD
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # BATCH-WISE TRAINING
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x[index * batch_size::], p[index * batch_size::])
                index = 0
            else:
                loss = self.model.train_on_batch(x[index * batch_size:(index + 1) * batch_size],
                                                 p[index * batch_size:(index + 1) * batch_size])
                index += 1

            # SAVE INTERMEDIATE MODEL
            if ite % save_interval == 0:
                print('The intermediate model is saved to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')

        logdict = dict(iter=ite, sh_score=sh_score, ch_score=ch_score, db_score=db_score, loss=loss)
        print("Cluster count: ", pd.Series(y_pred).unique().size)
        logwriter.writerow(logdict)

        # SAVE THE FINAL MODEL
        logfile.close()
        print('Saving the final model to:', save_dir + '/DEC_final_model.h5')
        self.model.save_weights(save_dir + '/DEC_final_model.h5')
        return y_pred
