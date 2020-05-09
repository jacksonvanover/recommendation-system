env:
	docker run --rm -it -v ${PWD}:/root/ -w /root/ jacksonvanover/recommendation-system:env 

clean:
	rm -f data/*.pckl data/my_* data/spectralClustering*.csv data/matrixCompletion*.csv jacksonvanover_preds*