from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from glob import glob

from data_generator import LineIterator
from tokenizers import tokenize_regex
from gensim_word2vec import GensimWord2Vec
from config import PRC_DIR, MOD_DIR

import sys
import os


def _parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name',
                        type=str,
                        help='Name to use for saving the word2vec model')
    parser.add_argument('--train_data',
                        type=str,
                        default=PRC_DIR,
                        help="Training data.")
    parser.add_argument('--save_path',
                        type=str,
                        default=MOD_DIR,
                        help="Directory to write the model.")
    parser.add_argument('--min_count',
                        type=int,
                        default=3,
                        help=("The minimum number of word occurrences for it to be "
                              "included in the vocabulary."))
    parser.add_argument('--window_size',
                        type=int,
                        default=5,
                        help=("The number of words to predict to the left and right "
                              "of the target word."))
    parser.add_argument('--neg_samples',
                        type=int,
                        default=8,
                        help="Negative samples per training example.")
    parser.add_argument('--subsample',
                        type=float,
                        default=0.00001,
                        help=("Subsample threshold for word occurrence. Words that appear "
                              "with higher frequency will be randomly down-sampled. Set "
                              "to 0 to disable."))
    parser.add_argument('--load',
                        type=str,
                        default=None,
                        help='Name of the word2vec model to load. Should be in MOD_DIR.')
    parser.add_argument('--phraser_load',
                        type=str,
                        default=None,
                        help='Name of the phraser model to load. Should be in MOD_DIR.')
    parser.add_argument('--workers',
                        type=int,
                        default=8,
                        help='Number of worker threads to use for training the model.')
    parser.add_argument('--epochs',
                        type=int,
                        default=25,
                        help='Number of training epochs.')
    parser.add_argument('--phraser',
                        default=False,
                        action='store_true',
                        help='Toogle automatic collocation detection')
    parser.add_argument('--phraser_th',
                        type=int,
                        default=100,
                        help='Occurance thresold for bigrams to be considered collocations.')
    parser.add_argument('--dim',
                        type=int,
                        default=300,
                        help='Word2Vec space dimensionality.')
    parser.add_argument('--hsoftmax',
                        default=False,
                        action='store_true',
                        help='Toogle hierarchical softmax.')
    parser.add_argument('--voc_size',
                        type=int,
                        default=1000000,
                        help='Maximum vocabulary size')
    parser.add_argument('--skipgram',
                        default=False,
                        action='store_true',
                        help='Toogle skip-gram word2vec model (otherwise CBOW is used).')

    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    args, model_args = _parse_args()

    if not args.name:
        print("Model --name not specified, exiting.")
        sys.exit(1)

    print("Using training data from {}".format(args.train_data))

    inpaths = glob(os.path.join(args.train_data, "*"))

    litr = LineIterator(inpaths, tokenizer=tokenize_regex, progress=True, lower=True)

    w2v_name = os.path.join(args.save_path, args.name)
    phr_name = w2v_name + '.phraser'

    if args.load:
        w2v_load = os.path.join(args.save_path, args.load)
    else:
        w2v_load = None

    if args.phraser_load:
        phr_load = os.path.join(args.save_path, args.phraser_load)
    else:
        phr_load = None

    md = GensimWord2Vec(litr,
                        save_to=w2v_name,
                        save_phraser_to=phr_name,
                        load_from=w2v_load,
                        load_phraser_from=phr_load,
                        phrasing=args.phraser,
                        phrase_threshold=args.phraser_th,
                        workers=args.workers,
                        size=args.dim,
                        hsoftmax=args.hsoftmax,
                        window=args.window_size,
                        min_count=args.min_count,
                        max_vocab_size=args.voc_size,
                        skip_gram=args.skipgram,
                        downsample=args.subsample,
                        negative_samples=args.neg_samples)

    
    
    
    md.train(args.epochs)


if __name__ == "__main__":
    print(os.getcwd())
    main()
