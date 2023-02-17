class RemoveDuplicates:
    def __init__(self, df: pd.DataFrame) -> None:
        self.__df = df
        self.__vectorizer = TfidfVectorizer(analyzer='word',
                                            ngram_range=(2, 5),
                                            stop_words='english',
                                            max_features=5000)

    def __combine_text(self) -> list[str]:
        """
        this function takes 'structured_data' column and converts it into list of string
        :return: list of string of sentences
        """
        self.__corpus = '.'.join(self.__df['structured_data'].to_list())
        return self.__corpus.split(".")

    def train(self) -> TfidfVectorizer:
        """
        this function, fits and transform tfidf vectorizer using the data under 'structured_data' column
        :return: vectors generated on the basis of data present under 'structured_data' column
        """
        self.__corpus = self.__combine_text()  # corpus: collection of texts
        self.__vectorizer.fit_transform(self.__corpus)
        return self.__vectorizer

    def __get_sentence_embeddings(self, vectorizer, row_no: int, no_of_sentences: int) -> list:
        """
        this function returns indices of sentence that are repeated many times throughout the DataFrame
        :param row_no: indicates the DataFrame's index on which processing is to be done
        :param no_of_sentences: indicates the number of sentences present in DataFrame's index specified above
        :return: indices of sentences within the DataFrame's index that are repeated many times
        """
        self.__vectorizer = vectorizer
        structured_data = self.__df['structured_data'][row_no].split(".")
        cleaned_data = self.__df['clean_data'][row_no].split(".")
        
        X_test = [[np.exp(np.sqrt(self.__vectorizer.transform(structured_data)[i].toarray())).mean(), i, row_no, cleaned_data[i]] for i in range(no_of_sentences)]
        X_test.sort()
        return X_test

    def embeddings_for_sentence(self, vectorizer) -> list:
        """
        this function returns the sentence that are removed from the data.
        :return: sentences removed from the data.
        """
        __row = []
        for i in range(self.__df.shape[0]):
            structured_data = self.__df['structured_data'][i].split(".")
            cleaned_data = self.__df['clean_data'][i].split(".")
            no_of_sentences = len(structured_data)
            sentence_embeddings = self.__get_sentence_embeddings(vectorizer, i, no_of_sentences)
            __row.append(sentence_embeddings)
        return __row
    
    def __compute_threshold(self) -> float:
        """
        this function computes appropriate threshold
        :return: returns the calculated threshold
        """
        if "sentence_embeddings" in self.__df.columns:
            lst = list(itertools.chain(*self.__df['sentence_embeddings'].tolist()))
            sentence_embeddings = [i for (i, j, k, l) in lst]
            sentence_embeddings.sort()
            thresh = np.percentile(sentence_embeddings, 15, interpolation = 'midpoint')
            Q1 = np.percentile(sentence_embeddings, 25, interpolation = 'midpoint') 
            Q2 = np.percentile(sentence_embeddings, 50, interpolation = 'midpoint') 
            Q3 = np.percentile(sentence_embeddings, 75, interpolation = 'midpoint') 
            IQR = Q3 - Q1
            threshold = Q1 - 1.5 * IQR 
            return Q1
    
    def duplicate_sentence_indices(self) -> list:
        """
        this function gives the indices of the duplicate sentences
        :return: returns the list duplicate sentence indices
        """
        __all_rows = []
        threshold = self.__compute_threshold()
        for i in self.__df['sentence_embeddings']:
            __row = []
            for [j, k, l, m] in i:
                if 0 < j < threshold:
                    __row.append([j, k, l, m])
            __all_rows.append(__row)
        return __all_rows
    
    def duplicate_sentences(self) -> list:
        """
        this function returns list of duplicate sentences
        :return: returns duplicate sentence's list
        """
        __all_rows = []
        for i in range(self.__df.shape[0]):
            cleaned_data = self.__df['clean_data'][i].split(".")
            duplicate_sentence_indices = self.__df['duplicate_sentence_indices'][i]
            __rows = [cleaned_data[k] for [j, k, l, m] in duplicate_sentence_indices]
            __all_rows.append('.'.join(__rows))
        return __all_rows
