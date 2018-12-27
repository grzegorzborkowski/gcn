### DBPL dataset parser, and preprocessor for creating a graph of graph. 

#### Usage
```
usage: DBLPPipeline.py [-h] [--authors_count AUTHORS_COUNT]
                       [--words_min_frequency WORDS_MIN_FREQUENCY]
                       [--min_edges_for_article MIN_EDGES_FOR_ARTICLE]
                       [--ngrams NGRAMS] [--debug] [--no-debug]

optional arguments:
  -h, --help            show this help message and exit
  --authors_count AUTHORS_COUNT
  --words_min_frequency WORDS_MIN_FREQUENCY
  --min_edges_for_article MIN_EDGES_FOR_ARTICLE
  --ngrams NGRAMS
  --debug
  --no-debug
```
#### Arguments
--authors_count - how many top authors (in terms of number of articles written) will be kept. All articles not written by those actors will be discarded. <br>
--words_min_frequency - at least how many times a word has to occur in abstracts not to be removed
<br>
--min_edges_for_article - remove all articles occuring less times than this number
--ngrams - how many adjacent words to use in creating internal graphs. Default set to 3.
--debug/no-debug enable or disable printing debug info. Default set to debug.

#### Extra information
The dataset is here: 
https://aminer.org/lab-datasets/citation/DBLP-citation-Jan8.tar.bz2

The transformation of the dataset are described:
https://openreview.net/pdf?id=ryeaZhRqFm (page 13)


