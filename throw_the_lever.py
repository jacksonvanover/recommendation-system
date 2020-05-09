#! /usr/bin/env python3

import argparse

from src.spectral_clustering_solution import ClusteringRecommender
from src.matrix_completion_solution import MatrixCompletionRecommender


# Initialize CLI argument parser
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--method', metavar='chosen-method', type=str, required=True, dest='method',
                    help='The chosen recommendation method. One of "spectral-clustering" or "matrix-completion.'
                    )
parser.add_argument('--splitter', metavar='chosen-splitter-method', type=str, required=False, dest='splitter', default=None,
                    help='The chosen splitting method for cross-file validation. One of "my-splitter", "sklearn", or "preprocessed". If not provided, no cross-file validation will occur and predictions for test.csv will be output.'
                    )
args = parser.parse_args()


def main():
    if not args.splitter:
        production = True
    else:
        production = False

    print("\n====== INITIALIZING RECOMMENDERS ======\n")
    if args.method == "spectral-clustering":
        recommender = ClusteringRecommender(production=production, splitter=args.splitter)
    elif args.method == "matrix-completion":
        recommender = MatrixCompletionRecommender(production=production, splitter=args.splitter)

    print("\n====== PERFORMING CLUSTERING ======\n")
    recommender.cluster()

    print("\n====== MAKING RECOMMENDATIONS ======\n")
    recommender.recommend()

    if not production:
        print("\n====== PERFORMING CROSS-VALIDATION ======\n")
        error = recommender.calculate_error()
        print("\tCorrect: {}/{} = {}%".format(error[0], error[1], error[0]/error[1] * 100))
        print("\tMSE: {}\n".format(error[2]))


if __name__ == "__main__":
    main()