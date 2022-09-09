from data_modified import ProcessData
import os
from time import time
from deep_embedded_clustering import DEC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# DATA LOADING
data_obj = ProcessData()
csv_data = ProcessData.read_data()
df_scaled = ProcessData.normalise_data(csv_data)
X = df_scaled.copy()
print(X.shape)

# HYPER_PARAMETERS
init = 'glorot_uniform'
pretrain_optimizer = 'adam'
pretrain_epochs = 100
batch_size = 128
update_interval = 50
tol = 0.00001
max_iter = 1000
dims = [X.shape[-1], 20, 500, 50, 5]
n_clusters = 2
autoencoder_weights = None# or specify a file #"./results/weights.h5"

# CREATE FOLDER TO SAVE RESULTS
results_folder = './results'
if not os.path.exists(results_folder):
    os.mkdir(results_folder)

# BASELINE KMEANS
km = KMeans(init="k-means++", n_clusters=2)
t0 = time()
y_pred = km.fit_predict(X)
sh_score = silhouette_score(X, y_pred)
ch_score = calinski_harabasz_score(X, y_pred)
db_score = davies_bouldin_score(X, y_pred)
print('Baseline Kmeans sh_score: %.4f,  ch_score: %.4f,  db_score: %.4f====>' % (sh_score, ch_score, db_score))
print('clustering time: ', (time() - t0))

# PRETRAINING
print(dims)
dec = DEC(dims=dims, n_clusters=n_clusters, init=init)
if autoencoder_weights is None:
    dec.pretrain(x=X, y=X, optimizer=pretrain_optimizer,
                 epochs=pretrain_epochs, batch_size=batch_size,
                 save_dir=results_folder)
else:
    dec.autoencoder.load_weights(autoencoder_weights)
dec.model.summary()

# DEEP CLUSTERING
t0 = time()
dec.compile(optimizer="adam", loss='kld')
y_pred = dec.fit(X, y=X, tol=tol, maxiter=max_iter, batch_size=batch_size,
                 update_interval=update_interval, save_dir=results_folder)
print('y_pred:', y_pred)
print('clustering time: ', (time() - t0))

# SAVE CLUSTER LABELS
csv_data["cluster"] = y_pred
csv_data.to_csv("./results/output.csv", index=False)
