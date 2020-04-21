import numpy as np
import pandas as pd
import os
from scipy.sparse import csc_matrix
from scipy.linalg import svd
from sklearn.impute import SimpleImputer
from split_training_set import split_training_set

# set CWD to root of project directory tree
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

split_training_set()

# load dataset
df_filled = pd.read_csv("./data/my_train.csv")
df_blank = pd.read_csv("./data/my_test_blank.csv")

# convert all question marks to zeros
df_blank["rating"] = df_blank["rating"].apply(lambda x : 0)

# merge the two
df = pd.concat([df_filled,df_blank], ignore_index=True, sort=False).sort_values(by=['customer-id', 'movie-id']).reset_index(drop=True)

# dicts for mapping ids to indices in the customer_nodes matrix
cid_to_index = {key:value for value, key in enumerate(df["customer-id"].unique())}
mid_to_index = {key:value for value, key in enumerate(df["movie-id"].unique())}

# construct customer_nodes matrix
row = np.array(df['customer-id'].apply(lambda x : cid_to_index[x]).tolist())
col = np.array(df['movie-id'].apply(lambda x : mid_to_index[x]).tolist())
data = np.array(df['rating'].tolist())
customer_nodes = csc_matrix( (data,(row,col)), shape=(len(cid_to_index), len(mid_to_index))).todense()

my_imputer = SimpleImputer(missing_values=0, strategy='mean')
my_imputer.fit_transform(customer_nodes)

U, s, V = svd(customer_nodes)

import pickle
with open("U.pckl", "wb") as f:
    pickle.dump(U, f)

with open("V.pckl", "wb") as f:
    pickle.dump(V, f)

with open("s.pckl", "wb") as f:
    pickle.dump(s, f)
