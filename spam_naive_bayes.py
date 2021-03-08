class Spam_Naive_Bayes(object):
    """Implementation of Naive Bayes for Spam detection."""
    def __init__(self):
        self.num_messages = dict()
        self.word_counts = dict()
        self.class_priors = dict()

    def clean(self, s):
        translator = str.maketrans("", "", string.punctuation)
        return s.translate(translator)

    def tokenize(self, text):
        text = self.clean(text).lower()
        return re.split("\W+", text)

    @staticmethod
    def get_word_counts(words):
        """
        Generate a dictionary 'word_counts' 
        Hint: You can use helper function self.clean and self.toeknize.
              self.tokenize(x) can generate a list of words in an email x.

        Inputs:
            -words : list of words that is used in a data sample
        Output:
            -word_counts : contains each word as a key and number of that word is used from input words.
        """

        word_counts = dict()
        for word in words:
            if word in word_counts.keys():
                word_counts[word] += 1
            else:
                word_counts.update({word: 1})

        return word_counts

    def fit(self, X_train, y_train):
        """
        compute likelihood of all words given a class

        Inputs:
            -X_train : list of emails
            -y_train : list of target label (spam : 1, non-spam : 0)
            
        Variables:
            -self.num_messages : dictionary contains number of data that is spam or not
            -self.word_counts : dictionary counts the number of certain word in class 'spam' and 'ham'.
            -self.class_priors : dictionary of prior probability of class 'spam' and 'ham'.
        Output:
            None
        """
        # calculate naive bayes probability of each class of input x

        # todo, wow this is slow
        counts = []
        # clean up and get word counts
        vocab = []
        for mail in X_train:
            tmp = self.tokenize(mail)
            for word in tmp:
                if word not in vocab:
                    vocab.append(word)
            count = self.get_word_counts(tmp)
            counts.append(count)

        n_1 = 0
        n_0 = 0
        word_0 = dict()
        word_1 = dict()
        num_unique = len(vocab)
        for mail, classifier in zip(counts, y_train):
            for word in mail:
                if classifier:
                    n_1 += mail[word]
                    word_1.update({word: mail[word]})
                else:
                    n_0 += mail[word]
                    word_0.update({word: mail[word]})

        parameters_1 = {word: 0 for word in vocab}
        parameters_0 = {word: 0 for word in vocab}
        alpha = 1
        for mail, classifier in zip(counts, y_train):
            for word in mail:
                if classifier:
                    class1_n = mail[word]
                    p_word_1 = (class1_n + alpha) / (n_1 + alpha * num_unique)
                    parameters_1[word] = p_word_1
                else:
                    class0_n = mail[word]
                    p_word_0 = (class0_n + alpha) / (n_0 + alpha * num_unique)
                    parameters_0[word] = p_word_0

        self.class_priors = {'spam': len([1 for y in y_train if y == 1]) / len(y_train), 'ham': len([0 for y in y_train if y == 0]) / len(y_train)}
        self.num_messages = {'spam': parameters_1, 'ham': parameters_0}
        self.word_counts = {'spam': word_1, 'ham': word_0}

    def predict(self, X):
        """
        predict that input X is spam of not. 
        Given a set of words {x_i}, for x_i in an email(x), if the likelihood 
        
        p(x_0|spam) * p(x_1|spam) * ... * p(x_n|spam) * y(spam) > p(x_0|ham) * p(x_1|ham) * ... * p(x_n|ham) * y(ham),
        
        then, the email would be spam.

        Inputs:
            -X : list of emails

        Output:
            -result : A numpy array of shape (N,). It should tell rather a mail is spam(1) or not(0).
        """

        result = []
        for x in X:
            # calculate naive bayes probability of each class of input x
            p_spam = self.class_priors['spam']
            p_ham = self.class_priors['ham']
            for word in x:
                if word in self.word_counts['spam']:
                    p_spam *= self.num_messages['spam'][word]

                if word in self.word_counts['ham']:
                    p_ham *= self.num_messages['ham'][word]

            if p_spam > p_ham:
                result.append(1)
            else:
                result.append(0)

        result = np.array(result)
        return result
