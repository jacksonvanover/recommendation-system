import os
import pickle
import pandas as pd
import numpy as np

from scipy.sparse import csc_matrix
from scipy.linalg import svd, sqrtm

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

from split_training_set import split_training_set


# set CWD to root of project directory tree
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")


class MatrixCompletionRecommender():
    
    def __init__(self, production=True, splitter="my-splitter"):

        self.production = production
        self.splitter = splitter

        if self.production:
            self.df_filled = pd.read_csv("./data/train.csv")
            self.df_blank = pd.read_csv("./data/test.csv")

        else:
            print("\n====== SPLITTING TRAINING SET ======\n")

            if splitter == "my-splitter":
                split_training_set()
            
                self.df_filled = pd.read_csv("./data/my_train.csv")
                self.df_blank = pd.read_csv("./data/my_test_blank.csv")
                self.df_answers = pd.read_csv("./data/my_test_answers.csv")

            elif splitter == "sklearn":
                split = train_test_split(pd.read_csv("./data/train.csv"))

                self.df_filled = split[0]
                self.df_answers = split[1]
                self.df_blank = self.df_answers.copy(deep=True)
                self.df_blank["rating"] = self.df_blank["rating"].apply(lambda x : "?")

            elif splitter == "preprocessed":
                self.df_filled = pd.read_csv("./data/my_train.csv")
                self.df_blank = pd.read_csv("./data/my_test_blank.csv")
                self.df_answers = pd.read_csv("./data/my_test_answers.csv")                

            else:
                raise Exception("Must specify a splitting method")
    
        # convert all question marks to zeros
        self.df_blank["rating"] = self.df_blank["rating"].apply(lambda x : 0)

        # merge the two
        self.df_combined = pd.concat([self.df_filled,self.df_blank], ignore_index=True, sort=False).sort_values(by=['customer-id', 'movie-id']).reset_index(drop=True)

        # dicts for mapping ids to indices in the customer_nodes matrix
        self.cid_to_index = {key:value for value, key in enumerate(self.df_combined["customer-id"].unique())}
        self.mid_to_index = {key:value for value, key in enumerate(self.df_combined["movie-id"].unique())}

        self.customer_nodes = self.build_customer_nodes_matrix()


    def build_customer_nodes_matrix(self):

        row = np.array(self.df_combined['customer-id'].apply(lambda x : self.cid_to_index[x]).tolist())
        col = np.array(self.df_combined['movie-id'].apply(lambda x : self.mid_to_index[x]).tolist())
        data = np.array(self.df_combined['rating'].tolist())
        customer_nodes = csc_matrix( (data,(row,col)), shape=(len(self.cid_to_index), len(self.mid_to_index))).todense()

        return customer_nodes


    def impute(self, matrix):
        matrix = np.where(matrix==0, np.nan, matrix)

        my_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        return my_imputer.fit_transform(matrix)


    def cluster(self, k=5):

        imputed_customer_nodes = self.impute(self.customer_nodes)

        if self.splitter == "preprocessed" and os.path.exists("./data/V.pckl"):
            with open("./data/U.pckl", "rb") as f:
                U = pickle.load(f)

            with open("./data/V.pckl", "rb") as f:
                V = pickle.load(f)

            with open("./data/s.pckl", "rb") as f:
                s = pickle.load(f)

        else:
            U, s, V = np.linalg.svd(imputed_customer_nodes, full_matrices=False)

            with open("./data/U.pckl", "wb") as f:
                pickle.dump(U, f)

            with open("./data/V.pckl", "wb") as f:
                pickle.dump(V, f)

            with open("./data/s.pckl", "wb") as f:
                pickle.dump(s, f)

        s = sqrtm(np.diag(s)[0:k, 0:k])
        U=U[:, 0:k]
        V=V[0:k, :]

        Us=np.dot(U,s)
        sV=np.dot(s,V)
        UsV = np.dot(Us, sV)

        self.recommender_matrix = UsV


    def estimate_rating(self, cid, mid):

        temp = round(self.recommender_matrix[self.cid_to_index[cid]][self.mid_to_index[mid]])

        return temp


    def recommend(self):
        # fill in estimated ratings
        self.df_blank["rating"] = self.df_blank.apply(lambda row : self.estimate_rating(row["customer-id"], row["movie-id"]), axis=1)
        
        # write out result
        with open("jacksonvanover_preds_matrix.txt", "w") as f:
            for rating in self.df_blank["rating"].tolist():
                print(int(rating), file=f)

        self.df_blank.to_csv("./data/matrixCompletion_test_filled.csv", index=False)


    def calculate_error(self):
        if self.production:
            raise Exception("No ground truth to evaluate against!")
        else:
            MSE = mean_squared_error(self.df_answers["rating"], self.df_blank["rating"])

            correct = 0
            total = 0
            for answer, guess in zip(self.df_answers["rating"].tolist(), self.df_blank["rating"]):
                total += 1
                if answer == guess:
                    correct += 1

            return (correct, total, MSE)