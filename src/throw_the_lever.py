#! /usr/bin/env python3

from spectral_clustering_solution import ClusteringRecommender


print("\n====== INITIALIZING RECOMMENDER ======\n")
recommender = ClusteringRecommender(production=False)

print("\n====== PERFORMING CLUSTERING ======\n")
recommender.cluster(n_clusters=25, n_components=7, n_iter=10, n_neighbors=500)

print("\n====== MAKING RECOMMENDATIONS ======\n")
recommender.recommend()

print(recommender.calculate_error())
recommender.print_stats_for_nerds()