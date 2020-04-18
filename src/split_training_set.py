import pandas as pd
import math
import os

# set CWD to root of project directory tree
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")


def split_training_set():
    """
    Splits the training set into a training set and an evaluation set.
    
    Inputs:
        None
    
    Outputs:
        Writes out three csv files to the data directory
    
    Function to split training set into a training set and an evaluation
    set. Guarantees that all movie-ids and customer-ids included in the
    evaluation set are still represented in the training set.
    """

    print("\n====== SPLITTING TRAINING SET ======\n")

    # load dataset
    df = pd.read_csv("./data/train.csv")
    df.sort_values(by=['customer-id'], inplace=True)

    # for each customer-id and each movie-id, figure out
    # how many entries constitute 10% of their total
    # representation in the whole dataset
    eval_size_per_cid = df['customer-id'].value_counts().apply(lambda x: math.floor(x/10)).to_dict()
    eval_size_per_mid = df['movie-id'].value_counts().apply(lambda x: math.floor(x/10)).to_dict()

    # split training set
    my_train = []
    my_eval = []
    for index, row in df.iterrows():
        if eval_size_per_cid[row["customer-id"]] != 0 and eval_size_per_mid[row["movie-id"]] != 0:
            my_eval.append(row)
            eval_size_per_cid[row["customer-id"]] -= 1
            eval_size_per_mid[row["movie-id"]] -= 1        
        else:
            my_train.append(row)

    # construct dataFrames and write them out as csv files
    df_my_train = pd.DataFrame(my_train)
    df_my_eval_answers = pd.DataFrame(my_eval)
    df_my_eval_blank = pd.DataFrame(my_eval)
    df_my_eval_blank["rating"] = df_my_eval_blank["rating"].apply(lambda x : "?")

    df_my_train.to_csv("./data/my_train.csv", index=False)
    df_my_eval_answers.to_csv("./data/my_test_answers.csv", index=False)
    df_my_eval_blank.to_csv("./data/my_test_blank.csv", index=False)

if __name__ == "__main__":
    split_training_set()