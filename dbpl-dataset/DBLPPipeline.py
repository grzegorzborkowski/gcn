import csv
import copy
import nltk
import collections
from operator import itemgetter
from nltk.corpus import stopwords

class DBLP():

    DEBUG = True

    def __init__(self):
        if DBLP.DEBUG: print ("[DBLP-Pipeline] Checking if stopwords for nltk package are downloaded")
        nltk.download('stopwords')
        if DBLP.DEBUG: print ("[DBLP-Pipeline] NLTK stopwords downloaded")
        self.summary = {}

    def read_and_filter_dataset(self):
        if DBLP.DEBUG: print("[DBLP-Pipeline] Reading articles from file")
        initial_articles = self.__read_articles__()
        self.summary['initial_articles_count'] = len(initial_articles)
        if DBLP.DEBUG: print("[DBLP-Pipeline] Reading articles from file has finished. Filtering articles without abstract")
        articles_without_abstract = self.__filter_articles_without_abstract__(initial_articles)
        if DBLP.DEBUG: print("[DBLP-Pipeline] Filtered articles without abstract. Tokenizing and removing stop words from abstract")
        articles_tokenized = self.__tokenize_and_remove_step_words_from_abstract__(articles_without_abstract)
        if DBLP.DEBUG: print("[DBLP-Pipeline] Finding most frequent authors")
        most_frequent_authors = self.__find__most_frequent_authors__(articles_tokenized)
        if DBLP.DEBUG: print("[DBLP-Pipeline] Filtering articles not containing top authors")
        articles_with_top_authors_only = self.__filter_articles_not_containing_top_authors__(articles_tokenized, most_frequent_authors)

        if DBLP.DEBUG: print("[DBLP-Pipeline] Calculating words frequency")
        top_words = self.__calculate_frequency_of_words__(articles_with_top_authors_only)
        if DBLP.DEBUG: print("[DBLP-Pipeline] Filtering unfrequent words from abstract")
        articles_with_filtered_abstract = self.__filter_unfrequent_words_from_abstract__(articles_with_top_authors_only, top_words)
        if DBLP.DEBUG: print("[DBLP-Pipeline] Removing non-existing artciles from quoted-by filed in articles")
        articles_with_updated_quoted_by = self.__update_quoted_by__(articles_with_filtered_abstract)
        self.summary['articles_count'] = len(articles_with_updated_quoted_by)
        return articles_with_updated_quoted_by

    def prepare_graph_of_graphs_from_articles(self):
        pass
    
    def write_summary_of_dataset(self):
        if DBLP.DEBUG: print("[DBLP-Pipeline] Writing a summary of dataset to summary.csv")
        with open("summary.csv", "w") as file:
            file.write("Initial articles count " + str(self.summary['initial_articles_count']) + "\n")
            file.write("Count of articles after transformations " + str(self.summary['articles_count']) + "\n")
            file.write("Number of words " + str(self.summary['number_of_words']) + "\n")
            file.write("Number of positive edges in graph " + str(self.summary['number_of_positive_edges']) + "\n")

    def __read_articles__(self):
        articles = []
        with open('DBLP-citation-Jan8.txt') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='#')
            for row in csv_reader:
                if len(row) >= 2:
                    data = row[1]
                    if len(data)<2:
                        continue
                    elif data[:1] == '*':
                        article = {}
                        article['quoted'] = []
                        article['title'] = data[1:]
                    elif data[:1] == '@':
                        article['authors'] = data[1:]
                    elif data[:1] == 't':
                        article['date'] = data[1:]
                    elif data[:1] == 'c':
                        article['source'] = data[1:]
                    elif data[:5] == 'index':
                        article['index'] = data[5:]
                        articles.append(article)
                    elif data[:1] == '%':
                        if 'quoted' in article:
                            article['quoted'].append(data[1:])
                        else:
                            article['quoted'].append(data[1:])
                    elif data[:1] == '!':
                        article['abstract'] = data[1:]
                    else:
                        continue
        return articles

    def __filter_articles_without_abstract__(self, articles):
        return [article for article in articles if 'abstract' in article and 'authors' in article]

    def __tokenize_and_remove_step_words_from_abstract__(self, articles):
        tokenizer = nltk.tokenize.SpaceTokenizer()
        stop_words_set = set(stopwords.words('english'))
        articles_copy = copy.deepcopy(articles)
        for article in articles_copy:
            try:
                article['abstract'] = tokenizer.tokenize(article['abstract'])
                article['abstract'] = [word for word in article['abstract'] if word not in stop_words_set]
            except AttributeError:
                pass
        return articles_copy

    def __find__most_frequent_authors__(self, articles, top=5000):
        authors_frequency = collections.defaultdict(int)
        for article in articles:
            authors = article['authors'].split(",")
            for author in authors:
                authors_frequency[author]+=1
        top_authors = list(sorted(authors_frequency.items(), key=itemgetter(1),reverse=True))
        top_authors = top_authors[:top]
        top_authors_without_occurences = [el[0] for el in top_authors]
        return top_authors_without_occurences

    def __filter_articles_not_containing_top_authors__(self, articles, top_authors):
        def all_authors_in_top(article, top_authors):
            for author in article['authors'].split(","):
                if author not in top_authors:
                    return False
            return True
        return [article for article in articles if all_authors_in_top(article, top_authors)]

    def __calculate_frequency_of_words__(self, articles, frequency_cap = 500):
        frequency = collections.defaultdict(int)
        for article in articles:
            for word in article['abstract']:
                frequency[word] += 1
            
        top_words = list(sorted(frequency.items(), key=itemgetter(1),reverse=True))
        top_words = [word[0] for word in top_words if word[1] > frequency_cap]
        self.summary['number_of_words'] = len(top_words)
        return top_words 

    def __filter_unfrequent_words_from_abstract__(self, articles, top_words):
        articles_copy = copy.deepcopy(articles)
        for article in articles_copy:
            article['abstract'] = [word for word in article['abstract'] if word in top_words]
        return articles_copy

    def __update_quoted_by__(self, articles):
        number_of_edges = 0 
        all_indexes = set()
        for article in articles:
            all_indexes.add(article['index'])
        articles_copy = copy.deepcopy(articles)
        for article in articles_copy:
            updated_quoted = [quoted for quoted in article['quoted'] if quoted in all_indexes]
            article['quoted'] = updated_quoted
            number_of_edges+= len(article['quoted'])
        self.summary['number_of_positive_edges'] = number_of_edges
        return articles_copy

if __name__ == "__main__":
    dblp = DBLP()
    filtered_documents = dblp.read_and_filter_dataset()
    dblp.write_summary_of_dataset()