from gensim.models.phrases import Phraser, Phrases
from gensim.models import Word2Vec


class GensimWord2Vec(object):
    def __init__(self, line_iterator, save_to, load_from=None,
                 save_phraser_to=None, load_phraser_from=None,
                 phrasing=True, phrase_threshold=10, delimiter=b'+',
                 size=256, window=5, workers=4,
                 skip_gram=False, hsoftmax=False, negative_samples=5,
                 min_count=5, max_vocab_size=None, downsample=0):

        """
        Initialize word2vec model

        Args:
        General
          line_iterator: data_generator.LineIterator instance
              Custom data source yielding on tokenized line of text at a time
          save_to: string
              Path where the word2vec model will be saved to
          load_from: string (default=None)
              Path where the word2vec model will be loaded from
          phrasing: bool (default=True)
              Toogle automatic detection of frequent word collocations
          phrase_threshold: int (default=10)
              Threshold for forming the phrases (higher means fewer phrases).
          delimiter: string (default=b'+')
              String that will be used to join collocating words into a token
          save_phraser_to: string (default=None)
              Path where the phraser model will be saved to
          load_phraser_from: string (default=None)
              Path where the phrases model will be loaded from
        Word2Vec-specific
          size: int (default=256)
            Word2Vec space dimensionality
          window: int (default=5)
            Maximum distance between the current and predicted word.
          workers: int (default=4)
            Number of worker threads used to train the model
          skip_gram: bool (default=False)
            Toogle skip-gram word2vec model. Otherwise, CBOW will be used
          hsoftmax: bool (default=False)
            Toogle hierarchical softmax. Useful for large vocabularies.
          negative_samples: int (default=5)
            Specifies how many noise words should be drawn
          min_count: int (default=5)
            Ignore all words with total frequency lower than this.
          max_vocab_size: int (default=None)
            If there are more unique words than this, prune the infrequent ones
          downsample: float (default=None, useful range is [0, 1e-5])
            Threshold for configuring which frequent words are downsampled
        """

        self.savepath = save_to
        self.model = Word2Vec(size=size, window=window,
                              min_count=min_count,
                              max_vocab_size=max_vocab_size,
                              sample=downsample, workers=workers,
                              sg=skip_gram, hs=hsoftmax,
                              negative=negative_samples, iter=1)

        if load_from:
            self.model = self.load_model(self.model, load_from, 'word2vec')

        self.line_iterator = line_iterator

        if phrasing:
            if load_phraser_from:
                self.phraser = self.load_model(Phraser,
                                               load_phraser_from, 'phraser')
            elif self.line_iterator is not None:
                self.phraser = self._train_phraser(min_count,
                                                   phrase_threshold,
                                                   delimiter)
            if save_phraser_to:
                self.save_model(self.phraser, save_phraser_to, 'phraser')
            if self.line_iterator is not None:
                self.line_iterator = self.phraser[self.line_iterator]

    def _train_phraser(self, min_count, phrase_threshold, delimiter):
        print("Training collocation detector...")
        return Phraser(Phrases(self.line_iterator,
                               min_count=min_count,
                               threshold=phrase_threshold,
                               delimiter=delimiter))

    def train(self, n_epochs):
        """
        Train word2vec model. Will build a vocab if the model doesn't have one yet.
        During training, the model will be saved to disk at the end of every epoch.

        Args:
          n_epochs: int
            Defines the number of epochs the model will be trained for.
        """

        if True:
            print("Building vocab...")
            self.model.build_vocab(self.line_iterator)
            #print("Built vocab of size {}".format(len(self.model.vocab)))

        self.model.train(self.line_iterator, self.line_iterator._count, epochs=n_epochs)
        self.save_model(self.model, self.savepath, 'word2vec')

        print("Finished training after {} epochs".format(n_epochs))

    def transform(self, tokens):
        """
        Get embeddings for an input sequence of tokens.
        If phraser is loaded, it will be used to preprocess the sequence.
        As a result, some pairs of tokens may be joined into one.
        OOV terms are completely ignored, no embedding will be returned for them.

        Args:
          tokens: list
            A list of tokens (strings) for which one needs to obtain embeddings
        Returns:
          wvecs: dict
            A dictionary where keys are tokens (or joint pairs of tokens)
            and values are corresponding word-embeddings.
        """
        if hasattr(self, 'phraser'):
            tokens = self.phraser[tokens]
        wvecs = {tok: self.model[tok] for tok in tokens if tok in self.model}
        return wvecs

    @staticmethod
    def save_model(model, path, mtype=''):
        model.save(path)
        print("Persisted {} model to {}".format(mtype, path))

    @staticmethod
    def load_model(model, path, mtype=''):
        model = model.load(path)
        print("Loaded {} model from {}".format(mtype, path))
        return model
