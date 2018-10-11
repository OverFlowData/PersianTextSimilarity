
""" I am using one hot vector for clustering with K-means couldn't upload my data but the code is valid
the code is wrriten by Samira Korani"""

import json
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans


"insert your file path"
with open('xxx.json', 'r') as data:
    data_1 = json.load(data)
    print(type(data_1))
    print(data_1.keys())
    data_2 = data_1['xxx']
     print(type(data_2))
    print(len(data_2))
    for data_3 in data_2[:1]:
         print(type(data_3))
        print(data_3.keys())
        data_4 = data_3['xxxx']
        print(data_4)
        print(type(data_4))
        data_5 =data_4.split()
        values = array(data_5)
        print(values)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)

scaler = StandardScaler().fit(onehot_encoded)
df_matrix=scaler.transform(onehot_encoded)
print(df_matrix)


km=MiniBatchKMeans(n_clusters=4,init='k-means++',max_iter=500,n_init=1000,init_size=1000,batch_size=1000,
                  verbose=False)
km_model=km.fit(onehot_encoded)
kmeanlabels=km.labels_
kmeanclusters = km.predict(onehot_encoded)
kmeandistances = km.transform(onehot_encoded)
