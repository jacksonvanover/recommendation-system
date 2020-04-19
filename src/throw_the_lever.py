#! /usr/bin/env python3

from spectral_clustering_solution import ClusteringRecommender


def test():
    print("\n====== INITIALIZING RECOMMENDER ======\n")
    recommender = ClusteringRecommender(production=False)

    clusters = [5,6,7,8]
    components = [3, 4, 5, 6]
    neighbors = [250, 500, 1000]

    configs = {}
    counter = 1
    for x in clusters:
        for y in components:
            for z in neighbors:
                
                print("=========")
                print("Testing config {}/60".format(counter))
                print("{} clusters, {} eigenvectors, {} nearest neighbors".format(x,y,z))

                recommender.cluster(n_clusters=x, n_components=y, n_iter=10, n_neighbors=z)
                recommender.recommend()

                error = recommender.calculate_error()
                print("\tMSE: {}\n".format(error))

                configs["{},{},{}".format(x,y,z)] = error
                counter += 1

    minimum = ("",10000)
    with open("output.txt", "w") as f:
        for config, error in configs.items():
            if error < minimum[1]:
                minimum = (config, error)
            print("{:>15} => {}".format(config, error))

    print(minimum)

def main():
    print("\n====== INITIALIZING RECOMMENDER ======\n")
    recommender = ClusteringRecommender(production=False)

    recommender.cluster(n_clusters=5, n_components=3, n_iter=10, n_neighbors=1000)
    recommender.recommend()

    error = recommender.calculate_error()
    print("\tMSE: {}\n".format(error))


if __name__ == "__main__":
    if sys.argv[1] == "test":
        test()
    else:
        main()