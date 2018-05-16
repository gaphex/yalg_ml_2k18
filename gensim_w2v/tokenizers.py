import re
import nltk
import pymystem3
from string import punctuation


url_rgx = re.compile('(https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,})')
re_tok = nltk.RegexpTokenizer("\w+")
mystem = pymystem3.Mystem()


def tokenize_split(text):
    return text.split()


def remove_pos_tags(text, delim="_"):
    return " ".join([t.rsplit(delim, 1)[0] for t in text.split()])


def tokenize_regex(text):
    return re_tok.tokenize(text)


def tokenize_mystem(text):
    text = text.replace("\x00", "")  # mystem hangs after null byte encounter

    tokens = []
    for token in mystem.analyze(text):
        if 'analysis' in token:
            if len(token['analysis']) > 0:
                lex = token['analysis'][0]['lex']
                tag = token['analysis'][0]['gr'].split('=')[0].split(',')[0]
            else:
                lex = token['text']
                tag = 'UNKN'

            new_token = '{}_{}'.format(lex, tag)
            tokens.append(new_token)
    return tokens


def tokenize_mystem_np(text):
    tokens = []
    for token in mystem.analyze(text):
        if 'analysis' in token:
            if len(token['analysis']) > 0:
                lex = token['analysis'][0]['lex']
            else:
                lex = token['text']
            tokens.append(lex)
    return tokens


def tokenize_sentences(text):
    return nltk.sent_tokenize(text)


def get_word_from_mystem(token):
    return token.rsplit("_", 1)[0]


def get_word_from_word(token):
    return token


def collapse_urls(s):
    def replace_punctuation(s, repl=""):
        return ''.join([c if c not in punctuation else repl for c in s.group(0)])
    return re.sub(url_rgx, replace_punctuation, s)
