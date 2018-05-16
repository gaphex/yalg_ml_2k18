import numpy as np
from tqdm import tqdm


class LineIterator(object):
    """
    As input accepts a list of filenames containing plain text,
    one sentence per line. If a string is passed as first argument,
    it will be treated as a list with one entry.

    Returns an iterator which returns tokenized sentences,
    one list of tokens at a time.

    Args:
      filenames: list or string
          Path(s) to plain-text files containing one sentence per line,
          to iterate through.
      tokenizer: callable (default=None)
          A function which takes a sentence string as input,
          and returns a list of it's tokens.
          If None (default), no tokenization will be performed.
      lower: bool (default=True)
          Toogle input lines lowering before tokenization.
      progress: bool (default=False)
          Toogle tqdm iteration progress display.
    """

    def __init__(self, filenames, tokenizer=None, lower=False, progress=False):
        self.filenames = np.atleast_1d(filenames)
        self._generator = self.line_generator(self.filenames)
        self._lower = lower
        self._prog = progress

        if tokenizer:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = lambda x: x

        if self._prog:
            self._count = self._count_lines()
            self._bar = tqdm(total=self._count)
            self._bar.clear()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            newline = next(self._generator)
            if self._prog:
                self._bar.update(1)
            return self._process_line(newline)
        except StopIteration:
            # This makes the iterator reusable
            self._generator = self.line_generator(self.filenames)
            if self._prog:
                self._bar.close()
                self._bar = tqdm(total=self._count)
                self._bar.clear()
            raise StopIteration

    def _count_lines(self):
        print("Enumerating data...")
        count = sum(1 for _ in tqdm(self.line_generator(self.filenames)))
        return count

    def _process_line(self, line):
        """
        All text processing routines are defined here.
        """
        line = line.strip()
        if self._lower:
            line = line.lower()
        tokenized = self._tokenizer(line)
        return tokenized

    @staticmethod
    def line_generator(filenames):
        for fname in filenames:
            with open(fname, 'r', encoding='utf-8') as f:
                for line in f:
                    yield line
