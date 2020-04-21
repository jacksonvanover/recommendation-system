import os
import pandas as pd
import numpy as np

from scipy.sparse import csc_matrix

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split

from split_training_set import split_training_set


# set CWD to root of project directory tree
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")


class ClusteringRecommender():

    def __init__(self, production=True, splitter="my-splitter"):

        self.production = production

        if self.production:
            self.df_train = pd.read_csv("./data/train.csv").sort_values(by=['customer-id','movie-id'])
            self.df_test = pd.read_csv("./data/test.csv")

        else:
            print("\n====== SPLITTING TRAINING SET ======\n")

            if splitter == "my-splitter":
                split_training_set()
            
                self.df_train = pd.read_csv("./data/my_train.csv").sort_values(by=['customer-id','movie-id'])
                self.df_test = pd.read_csv("./data/my_test_blank.csv")
                self.df_test_answers = pd.read_csv("./data/my_test_answers.csv")

            elif splitter == "sklearn":
                split = train_test_split(pd.read_csv("./data/train.csv"))

                self.df_train = split[0]
                self.df_test_answers = split[1]
                self.df_test = self.df_test_answers.copy(deep=True)
                self.df_test["rating"] = self.df_test["rating"].apply(lambda x : "?")

            elif splitter == "preprocessed":
                self.df_filled = pd.read_csv("./data/my_train.csv")
                self.df_blank = pd.read_csv("./data/my_test_blank.csv")
                self.df_answers = pd.read_csv("./data/my_test_answers.csv")

            else:
                raise Exception("Must specify a splitting method")
    
        # dicts for mapping ids to indices in the customer_nodes matrix
        self.cid_to_index = {key:value for value, key in enumerate(self.df_train["customer-id"].unique())}
        self.mid_to_index = {key:value for value, key in enumerate(self.df_train["movie-id"].unique())}

        self.customer_nodes = self.build_customer_nodes_matrix()

        self.affinity_matrix = self.build_affinity_matrix()


    def build_customer_nodes_matrix(self):

        row = np.array(self.df_train['customer-id'].apply(lambda x : self.cid_to_index[x]).tolist())
        col = np.array(self.df_train['movie-id'].apply(lambda x : self.mid_to_index[x]).tolist())
        data = np.array(self.df_train['rating'].tolist())
        customer_nodes = csc_matrix( (data,(row,col)), shape=(len(self.cid_to_index), len(self.mid_to_index)))

        return customer_nodes


    def build_affinity_matrix(self):

        affinity_matrix = cosine_similarity(self.customer_nodes)

        return affinity_matrix


    def cluster(self, n_clusters=5, n_components=3, n_iter=10, n_neighbors=1000):
        
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            n_components=n_components,
            random_state=25,
            n_init=n_iter,
            affinity='precomputed_nearest_neighbors',
            n_neighbors=n_neighbors
        ).fit(self.affinity_matrix)

        # create mapping from cid to cluster and vice-versa
        cid_to_cluster = { key:value for key,value in zip( self.df_train["customer-id"].unique() , clustering.labels_ ) }
        cluster_to_cids = [[] for x in range(n_clusters)]
        for cid, cluster_no in cid_to_cluster.items():
            cluster_to_cids[cluster_no].append(cid)

        self.cid_to_cluster = cid_to_cluster
        self.cluster_to_cids = cluster_to_cids


    def estimate_rating(self, cid, mid):

        # what other customers are in this cluster?
        try:
            cids = self.cluster_to_cids[self.cid_to_cluster[cid]]
            cid_indices = [self.cid_to_index[x] for x in cids]
        except KeyError:
            return 3

        try:
            mid_index = self.mid_to_index[mid]
        except KeyError:
            return 3      

        # get slice of cids from the same cluster for the particular mid
        temp = self.customer_nodes.getcol(mid_index).tocsr()[cid_indices]
        
        # eliminate zeros and calculate mean of nonzero data
        temp.eliminate_zeros()
        if temp.nnz > 0:
            average_rating = round(temp.data.mean())
        else:
            average_rating = 3

        return average_rating


    def recommend(self):

        # fill in estimated ratings
        self.df_test["rating"] = self.df_test.apply(lambda row : self.estimate_rating(row["customer-id"], row["movie-id"]), axis=1)
        
        # write out result
        with open("jacksonvanover_preds_clustering.txt", "w") as f:
            for rating in self.df_test["rating"].tolist():
                print(int(rating), file=f)

        df_test.to_csv("./data/my_test_filled.csv", index=False)

    def calculate_error(self):
        if self.production:
            raise Exception("No ground truth to evaluate against!")
        else:
            return mean_squared_error(self.df_test_answers["rating"], self.df_test["rating"])
            
        
    def print_stats_for_nerds(self):
        print("\n====== CLUSTER DISTRIBUTION ======\n")
        for cluster_no, cids in enumerate(self.cluster_to_cids):
            print("{} : {}".format(cluster_no, len(cids)))
