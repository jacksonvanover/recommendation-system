default: clean run

run:
	./src/throw_the_lever.py

clean:
	rm -f data/*.pckl data/my_* data/spectralClustering*.csv data/matrixCompletion*.csv jacksonvanover_preds*
