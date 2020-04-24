#! /usr/bin/env python3

from spectral_clustering_solution import ClusteringRecommender
from matrix_completion_solution import MatrixCompletionRecommender

def main():
    print("\n====== INITIALIZING RECOMMENDERS ======\n")
    recommender1 = MatrixCompletionRecommender(production=True)
    recommender2 = ClusteringRecommender(production=True)

    print("\n====== PERFORMING CLUSTERING ======\n")
    recommender1.cluster()
    recommender2.cluster()

    print("\n====== MAKING RECOMMENDATIONS ======\n")
    recommender1.recommend()
    recommender2.recommend()

    error = recommender1.calculate_error()
    print("Matrix Completion Solution:")
    print("\tCorrect: {}/{} = {}%".format(error[0], error[1], error[0]/error[1] * 100))
    print("\tMSE: {}\n".format(error[2]))

    error = recommender2.calculate_error()
    print("Spectral Clustering Solution:")
    print("\tCorrect: {}/{} = {}%".format(error[0], error[1], error[0]/error[1] * 100))
    print("\tMSE: {}\n".format(error[2]))


if __name__ == "__main__":
    main()
