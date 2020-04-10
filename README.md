# ECS271 – Unsupervised Learning with NetFlix Data

## Due:             Due 04/22/2019

## Worth:         20% of Final Grade

Read the instructions carefully, ask questions if you have any doubts. The training data sets are available in this document as links. Only use the data sets provided.

### Training Data
The training data set consists of 1 million transactions/records (user-movie pairs) of features/attributes:

- <movie id, customer id, rating, date recommended> as your training set as well as

- another data set of <movie id, date released, movie name>.

The feature types are `<customer id>` and `<movie id>` is an integer,
`<date recommended>`/`<date released>` is in date form,
`<recommendation>` is a integer from 1 to 5 and `<movie name>` is a
string.

### Pre-processing
You may discretize/aggregate features such as date if you so wish.

This data can be viewed/aggregated any number of ways:

This data is naturally a tensor (user versus movie versus time) with each entry being 0 (not watched) or 1 thru 5 (for the rating).

Alternatively, you can view the data as a bipartite (movie nodes and user nodes) evolving/multi-relational graph. You could also aggregate over time and create a simple 2-graph.


### Task
During the unsupervised module of the course we covered i) clustering methods and ii) matrix completion methods.

Your job is use one each of these approaches (or variation) and from the 1,000,000 records make predictions on the ratings for a further 250,000 movie-user pairs.

Please submit the following:

- For your clustering approach a) A short paragraph over-viewing the reasoning behind the approach, b) A diagrammatic summary of your approach, c) An at most one page detailed writeup on your approach including how you tuned any hyper-parameters. (15 points x 2)
- As above but for your matrix completion approach.
- For the test set on the website. Two separate  250,000 line file (named `<yourname_preds_clustering.txt>` and `<yourname_preds_matrix.txt>` with one prediction/score per line. The prediction must be an integer number b/w 1 and 5. I will use a quadratic loss function to score how accurate your approach is. (10 points x 2)
- A succint explanation regarding the differences between the predictions between the two methods and why they are different (5 bonus points).

For those of you interested in further reading, this type of problem is known as collaborative filtering and there are many papers, tutorials and surveys on the topic.