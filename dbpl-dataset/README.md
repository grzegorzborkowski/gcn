<h3>DBPL dataset parser, and preprocessor for creating a graph of graph. </h3>
```
usage: DBLPPipeline.py [-h] [--authors_count AUTHORS_COUNT]
                       [--words_min_frequency WORDS_MIN_FREQUENCY]
                       [--min_edges_for_article MIN_EDGES_FOR_ARTICLE]

optional arguments:
  -h, --help and exit
  --authors_count AUTHORS_COUNT
  --words_min_frequency WORDS_MIN_FREQUENCY
  --min_edges_for_article MIN_EDGES_FOR_ARTICL
```

--authors_count - how many top authors (in terms of number of articles written) will be kept. All articles not written by those actors will be discarded. <br>
--words_min_frequency - at least how many times a word has to occur in abstracts not to be removed
<br>
--min_edges_for_article - remove all articles occuring less times than this number


The dataset is here: 
https://aminer.org/lab-datasets/citation/DBLP-citation-Jan8.tar.bz2

The transformation of the dataset are described:
https://openreview.net/pdf?id=ryeaZhRqFm (page 13)


