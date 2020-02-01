
import argparse
import csv
import os

from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import ComplexityIndex, compute_indices
from rb.core.document import Document
from rb.core.lang import Lang
from rb.parser.spacy_parser import SpacyParser
from rb.processings.fluctuations.fluctuations import Fluctuations
from rb.processings.scoring.essay_scoring import EssayScoring
from rb.processings.train_models import (Preprocess, test_load_fast_text,
                                         test_load_lda, test_load_lsa,
                                         test_load_w2v, train_fast_text,
                                         train_lda, train_lsa, train_w2v,
                                         visualize_lda)
from rb.processings.readme_feedback.feedback import Feedback
from rb.processings.text_classifier.text_classifier import TextClassifier
from rb.processings.keywords.keywords_extractor import KeywordExtractor
from rb.processings.clustering.clustering import Clustering

from rb.similarity.vector_model import (CorporaEnum, VectorModel,
                                        VectorModelType)
from rb.similarity.vector_model_factory import create_vector_model
from rb.utils.rblogger import Logger
from rb.utils.utils import load_docs_all, str_to_lang, str_to_vmodel
from rb.utils.downloader import download_scoring, download_classifier

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

ff = open('debug2.txt', 'w', encoding='utf-8')

def do_scoring():
    global args, logger, test
    essay_scoring = EssayScoring()

    """ score is computed based on a predfined list of indices defined in rb/processings/scoring/*.txt)
        if you change the indicies in that list you have to reatrin the model (SVM)  """
    if args.scoring_actualize_indices:
        if args.scoring_lang is Lang.RO:
            model = create_vector_model(Lang.RO, VectorModelType.from_str('word2vec'), "readme")
        elif args.scoring_lang is Lang.EN:
            model = create_vector_model(Lang.EN, VectorModelType.from_str("word2vec"), "coca")
        else:
            logger.info(f'Unsopported lang {args.scoring_lang}')

        doc = Document(lang=args.indices_lang, text=test)
        cna_graph = CnaGraph(doc=doc, models=[model])
        compute_indices(doc=doc, cna_graph=cna_graph)
        indices = [repr(key) for key, _ in doc.indices.items()]
        with open(f'rb/processings/scoring/indices_{args.scoring_lang.value}_scoring.txt', 'wt', encoding='utf-8') as f:
            for ind in indices:
                f.write(ind + '\n')
    elif args.scoring_predict == False:
        essay_scoring.create_files_from_csv(path_to_csv_file='essays.csv', path_to_folder=args.scoring_base_folder)
        essay_scoring.compute_indices(base_folder=args.scoring_base_folder,
                                    write_file=args.scoring_indices_output_csv_file, 
                                    stats=args.scoring_stats_file, lang=args.scoring_lang, 
                                    nr_docs=None)
        results = essay_scoring.read_indices(base_folder=args.scoring_base_folder, 
                        path_to_csv_file=args.scoring_indices_output_csv_file)
        essay_scoring.train_svr(results, save_model_file=args.scoring_model_file)
    else: # just predict
        download_scoring(args.scoring_lang)
        score = essay_scoring.predict(test, file_to_svr_model=args.scoring_model_file)
        logger.info(f'Class for text {score}')

def do_classifier():
    global args, logger, test

    txt_class = TextClassifier()
    if args.classifier_predict == False:
        results = txt_class.read_indices()
        txt_class.train_svm(results, save_model_file=args.classifier_file)
    else:
        download_classifier(args.scoring_lang)
        class_text = txt_class.predict(test, file_to_svr_model=args.scoring_model_file)
        if class_text == 0:
            class_text = 'general'
        elif class_text == 1:
            class_text = 'literature'
        else:
            class_text = 'science'
        logger.info(f'Class of text is {class_text}')
        

def do_fluctuations():
    global args, logger, test
    fl = Fluctuations()
    return fl.compute_indices(test, lang=args.fluctuations_lang)


def do_train_model():
    global args, logger, test
    lang = args.train_lang
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
        test_load_lsa(path_lsa=os.path.join(args.train_base_folder, 'lsa.bin'))

    elif model is VectorModelType.WORD2VEC:
        train_w2v(sentences, args.train_base_folder)
        test_load_w2v(path_w2vec=os.path.join(args.train_base_folder, 'word2vec.model'), 
                      load_word2vec_format=False)
    else:
        logger.info('Model name not found')

def do_indices():
    if args.indices_lang is Lang.RO:
        model = create_vector_model(Lang.RO, VectorModelType.from_str('word2vec'), "readme")
    elif args.indices_lang is Lang.EN:
        model = create_vector_model(Lang.EN, VectorModelType.from_str("word2vec"), "coca")
    else:
        logger.info(f'No module for lang {args.indices_lang}')
        return

    all_rows, indices_abbr = [], []
    for i, pair_file_content in enumerate(load_docs_all(args.indices_base_folder)):
        filename = pair_file_content[0]
        content = pair_file_content[1]

        doc = Document(lang=args.indices_lang, text=content)
        """you can compute indices without the cna graph, but this means 
           some indices won't be computed"""
        cna_graph = CnaGraph(doc=doc, models=[model])
        compute_indices(doc=doc, cna_graph=cna_graph)
        
        if i == 0: # first row ith indices name
            for key, v in doc.indices.items():
                indices_abbr.append(repr(key))
            all_rows.append(['filename'] + indices_abbr)
        

        assert len(indices_abbr) == len(doc.indices.items()), 'wrong nr of indices'
        row = [filename]

        "TODO O(n * m) can be done in O(m)"

        for ind in indices_abbr:
            for key, v in doc.indices.items():
                if repr(key) == ind:
                    row.append(str(v))
                    print(ind, v, file=ff)
                    break
        all_rows.append(row)

        with open(os.path.join(args.indices_base_folder, 'stats.csv'), 'wt', encoding='utf-8') as stats_csv:
            csv_writer = csv.writer(stats_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerows(all_rows)


def do_feedback():
    feedback = Feedback()
    feedback.compute_extreme_values(path_to_csv='categories_readme/new_stats/stats_en.csv',
                                    output_file='readme_en.txt')
    feedback.compute_extreme_values(path_to_csv='categories_readme/new_stats/stats_general.csv',
                                    output_file='readme_general.txt')
    feedback.compute_extreme_values(path_to_csv='categories_readme/new_stats/stats_literature.csv',
                                    output_file='readme_literature.txt')
    feedback.compute_extreme_values(path_to_csv='categories_readme/new_stats/stats_science.csv',
                                    output_file='readme_science.txt')
""" TODO not finished for now """
def do_keywords():
    global args
    keywords_extractor = KeywordExtractor()
    keywords = keywords_extractor.extract_keywords(text=test, lang=args.keywords_lang)
    print(keywords)

def do_clustering():
    clustering = Clustering()
    clustering.compute_clustering()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run different tasks through this file')

    """parameters for scoring
       if you want to train: download essays.csv from nextcloud in Readerbench/corpora/RO/eseu_referate_ro and put in root folder
       and then run it to train an svm""" 
    parser.add_argument('--scoring', dest='scoring', action='store_true', default=False)
    parser.add_argument('--scoring_actualize_indices', dest='scoring_actualize_indices', action='store_true', default=False,
                        help="Change the indices used in scoring model. You have to retrain the model if you do so")
    parser.add_argument('--scoring_base_folder', dest='scoring_base_folder', action='store', default='essays_ro',
                        help='Base folder for files.')
    parser.add_argument('--scoring_predict', dest='scoring_predict', action='store_true', default=False)
    parser.add_argument('--scoring_indices_output_csv_file', dest='scoring_indices_output_csv_file',
                        action='store', default='measurements.csv',
                        help='Csv file for indices')
    parser.add_argument('--scoring_stats_file', dest='scoring_stats_file',
                        action='store', default='stats.csv',
                        help='Csv file with stats about files')
    parser.add_argument('--scoring_model_file', dest='scoring_model_file',
                        action='store', default='resources/ro/scoring/svr_gamma.p',
                        help='Pickle file for the model')
    parser.add_argument('--scoring_lang', dest='scoring_lang', default=Lang.RO.value, nargs='?', 
                        choices=[Lang.RO.value], help='Language for scoring (only ro supported)')

    """params for scoring"""
    parser.add_argument('--classifier', dest='classifier', action='store_true', default=False)
    parser.add_argument('--classifier_lang', dest='classifier_lang', default=Lang.RO.value, nargs='?', 
                        choices=[Lang.RO.value], help='Language for text classifier (only ro supported)')
    parser.add_argument('--classifier_predict', dest='classifier_predict', action='store_true', 
                         help='predict classifier')
    parser.add_argument('--classifier_file', dest='classifier_file', default='resources/ro/classifier/svr.p', 
                         help='classifier model')


    """parameters for fluctuations"""
    parser.add_argument('--fluctuations', dest='fluctuations', action='store_true', default=False)
    parser.add_argument('--fluctuations_lang', dest='fluctuations_lang', default=Lang.RO.value, nargs='?', 
                        choices=[Lang.RO.value, Lang.EN.value], help='Language for fluctuations (only ro and en supported)')

    """parameters for keywords"""
    parser.add_argument('--keywords', dest='keywords', action='store_true', default=False)
    parser.add_argument('--keywords_lang', dest='keywords_lang', default=Lang.RO.value, nargs='?', 
                        choices=[Lang.RO.value, Lang.EN.value], help='Language for keywords')
    """parameters for training models (LDA, LSA word2vec) 
       default parameters for training are good.
       TODO add parameters for training"""
    parser.add_argument('--train_model', dest='train_model', default='None', nargs='?',
                        choices=[VectorModelType.LDA.name, VectorModelType.LSA.name, VectorModelType.WORD2VEC.name, 'None'])
    parser.add_argument('--train_lang', dest='train_lang', default=Lang.RO.value, nargs='?', 
                        choices=[Lang.RO.value, Lang.ES.value, Lang.EN.value], 
                        help='Language for model')
    parser.add_argument('--train_base_folder', dest='train_base_folder', action='store', default='.',
                        help='Base folder for .txt files. Only files ended in .txt count')

    """compute indices
       by default we use only word2vec
       TODO add parameters for the other models"""
    parser.add_argument('--indices', dest='indices', action='store_true', default=False)
    parser.add_argument('--indices_lang', dest='indices_lang', default=Lang.RO.value, nargs='?', 
                        choices=[Lang.RO.value, Lang.EN.value], 
                        help='Language for indices')
    parser.add_argument('--indices_base_folder', dest='indices_base_folder', action='store', default='.',
                        help='Base folder for files. Only files ended in .txt count. Each file is considered a document')

    """ generate extreme values for indices"""
    parser.add_argument('--feedback_readme', dest='feedback_readme', action='store_true', default=False)

    parser.add_argument('--clustering_readme', dest='clustering_readme', action='store_true', default=False)
    
    
    args = parser.parse_args()
    args.scoring_lang: Lang = str_to_lang(args.scoring_lang)
    args.fluctuations_lang: Lang = str_to_lang(args.fluctuations_lang)
    args.train_lang = str_to_lang(args.train_lang)
    args.indices_lang = str_to_lang(args.indices_lang)
    args.keywords_lang = str_to_lang(args.keywords_lang)
    args.classifier_lang = str_to_lang(args.classifier_lang)

    for k in args.__dict__:
        if args.__dict__[k] is not None:
            logger.info('{} -> {}'.format(k,  str(args.__dict__[k])))

    if args.scoring:
        do_scoring()
    elif args.fluctuations:
        do_fluctuations()
    elif args.train_model != 'None':
        do_train_model()
    elif args.indices:
        do_indices()
    elif args.feedback_readme:
        do_feedback()
    elif args.classifier:
        do_classifier()
    elif args.keywords:
        do_keywords()
    elif args.clustering_readme:
        do_clustering()

