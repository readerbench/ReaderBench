
import argparse
import os
from rb.core.lang import Lang
from rb.utils.utils import str_to_lang
from rb.utils.rblogger import Logger
from rb.processings.scoring.essay_scoring import EssayScoring
from rb.processings.fluctuations.fluctuations import Fluctuations
from rb.similarity.vector_model import VectorModelType
from rb.processings.train_models import (Preprocess, train_w2v, test_load_w2v, train_fast_text,
             test_load_fast_text, train_lda, test_load_lda, visualize_lda, train_lsa, test_load_lsa)
from rb.utils.utils import str_to_lang, str_to_vmodel
from rb.parser.spacy_parser import SpacyParser

logger = Logger.get_logger()

test = """I. Evaluarea performantelor profesionale
        Motto:
        "Daca nu poti masura ceva, nu-l poti intelege,
        Daca nu-l poti intelege, nu-l poti controla,
        Daca nu-l poti controla, nu-l poti imbunatati."
        H.I.Harrington
        Capacitatea de cunoastere a omului este inseparabila de abilitatea sa de a evalua, care face posibila selectia, ierarhizarea si sistematizarea informatiilor. Fie ca vizeaza obiective, fenomene sau procese, plasarea pe o anumita pozitie a unei scari valorice determina in mod curent atitudini, decizii si actiuni.
        Din perspectiva manageriala, aprecierea rezultatelor unei activitati, raportate la obiectivele organizatiei si in relatie cu contextul real in care se defasoara, constituie o conditie a oricarui demers de perfectionare sau de adaptare. La nivel individual, de organizatie sau sistem, evaluarea corect efectuata permite intelegerea clara a deficientelor si deschide calea unor posibile imbunatatiri.1
        I.1. Tipuri de evaluare
        Evaluarea ca notiune are un caracter complex, determinat de diversi factori functie de care se realizeaza. Astfel, unii autori2 considera evaluarea personalului ca fiind"actul prin care un responsabil ierarhic efectueaza o apreciere formalizata a subordonatilor sai", iar daca aceste aprecieri au loc anual atunci este vorba despre"un sistem de apreciere" al organizatiei. Alti autori fac referire doar la performantele obtinute, fiind considerata"activitatea de baza a managementului resurselor umane, desfasurata in vederea determinarii gradului in care angajatii unei organizatii indeplinesc eficient sarcinile si responsabilitatile care le revin"3
"""

def do_scoring():
    global args, logger, test
    essay_scoring = EssayScoring()
    #essay_scoring.create_files_from_csv(path_to_csv_file='essays.csv', path_to_folder=args.base_folder)

    essay_scoring.compute_indices(base_folder=args.scoring_base_folder,
                                  write_file=args.scoring_indices_output_csv_file, 
                                  stats=args.stats_file, lang=args.scoring_lang, 
                                  nr_docs=None)
    results = essay_scoring.read_indices(base_folder=args.scoring_base_folder, 
                    path_to_csv_file=args.scoring_indices_output_csv_file)
    essay_scoring.train_svr(results, save_model_file=args.model_file)
    essay_scoring.predict(test, file_to_svr_model=args.model_file)


def do_fluctuations():
    global args, logger, test
    fl = Fluctuations()
    return fl.compute_indices(test, lang=args.fluctuations_lang)


def do_train_model():
    global args, logger, test
    lang = str_to_lang(args.train_lang)
    model = str_to_vmodel(args.train_model)
    parser = SpacyParser.get_instance()
    logger.info("Loading dataset from {}".format(args.train_base_folder))
    sentences = Preprocess(parser=parser, folder=args.train_base_folder,
                           lang=lang, split_sent=False, only_dict_words=False)

    if model is VectorModelType.LDA:
        train_lda(sentences, args.train_base_folder)
        test_load_lda(path_to_model=os.path.join(args.train_base_folder, 'lda.model'))
        
    elif model is VectorModelType.LSA:
        train_lsa(sentences, args.train_base_folder)
        test_load_lsa(path_lsa=os.path.join(args.train_model, 'lsa.bin'))

    elif model is VectorModelType.WORD2VEC:
        train_w2v(sentences, args.train_base_folder)
        test_load_w2v(path_w2vec=os.path.join(args.train_base_folder, 'word2vec.model'), 
                      load_word2vec_format=False)
    else:
        logger.info('Model name not found')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run different tasks through this file')

    """parameters for scoring""" 
    parser.add_argument('--scoring', dest='scoring', action='store_true', default=False)
    parser.add_argument('--scoring_base_folder', dest='scoring_base_folder', action='store', default='essays_ro',
                        help='Base folder for files.')
    parser.add_argument('--scoring_indices_output_csv_file', dest='scoring_indices_output_csv_file',
                        action='store', default='measurements.csv',
                        help='Csv file for with indices.')
    parser.add_argument('--stats_file', dest='stats_file',
                        action='store', default='stats.csv',
                        help='Csv file with stats about files.')
    parser.add_argument('--model_file', dest='model_file',
                        action='store', default='svr_gamma.p',
                        help='Pickle file for the model')
    parser.add_argument('--scoring_lang', dest='scoring_lang', default=Lang.RO.value, nargs='?', 
                        choices=[Lang.RO.value], help='Language for scoring (only ro supported)')

    """parameters for fluctuations"""
    parser.add_argument('--fluctuations', dest='fluctuations', action='store_true', default=False)
    parser.add_argument('--fluctuations_lang', dest='fluctuations_lang', default=Lang.RO.value, nargs='?', 
                        choices=[Lang.RO.value, Lang.EN.value], help='Language for fluctuations (only ro and en supported)')

    """parameters for training models (LDA, LSA word2vec) """
    parser.add_argument('--train_model', dest='train_model', default='None', nargs='?',
                        choices=[VectorModelType.LDA.name, VectorModelType.LSA.name, VectorModelType.WORD2VEC.name, 'None'])
    parser.add_argument('--train_lang', dest='train_lang', default=Lang.RO.value, nargs='?', 
                        choices=[Lang.RO.value, Lang.EN.value], 
                        help='Language for model')
    parser.add_argument('--train_base_folder', dest='train_base_folder', action='store', default='.',
                        help='Base folder for files.')

    args = parser.parse_args()
    args.scoring_lang: Lang = str_to_lang(args.scoring_lang)
    args.fluctuations_lang: Lang = str_to_lang(args.fluctuations_lang)

    for k in args.__dict__:
        if args.__dict__[k] is not None:
            logger.info('{} -> {}'.format(k,  str(args.__dict__[k])))

    if args.scoring:
        do_scoring()
    elif args.fluctuations:
        do_fluctuations()
    elif args.train_model != 'None':
        do_train_model()
