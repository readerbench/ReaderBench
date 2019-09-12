
import argparse
from rb.core.lang import Lang
from rb.utils.utils import str_to_lang
from rb.utils.rblogger import Logger
from rb.processings.scoring.essay_scoring import EssayScoring
from rb.processings.fluctuations.fluctuations import Fluctuations

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
    essay_scoring.compute_indices(csv_file_in=args.scoring_input_csv_file, lang=args.scoring_lang, 
                                  write_file=args.scoring_indices_output_csv_file, nr_docs=12)
    results = essay_scoring.read_indices(path_to_csv_file=args.scoring_indices_output_csv_file)
    essay_scoring.train_svr(results, save_model_file=args.scoring_svr_rbf_model_file)
    essay_scoring.predict(test, file_to_svr_model=args.scoring_svr_rbf_model_file)


def do_fluctuations():
    global args, logger, test
    fl = Fluctuations()
    return fl.compute_indices(test, lang=args.fluctuations_lang)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run different tasks through this file')

    """parameters for scoring""" 
    parser.add_argument('--scoring', dest='scoring', action='store_true', default=False)
    parser.add_argument('--scoring_input_csv_file', dest='scoring_input_csv_file', action='store', default='essays.csv',
                        help='File with corpus for scoring.')
    parser.add_argument('--scoring_lang', dest='scoring_lang', default=Lang.RO.value, nargs='?', 
                        choices=[Lang.RO.value], help='Language for scoring (only ro supported)')
    parser.add_argument('--scoring_indices_output_csv_file', dest='scoring_indices_output_csv_file', action='store', 
                        default='essays_eval_all.csv', help='Name of csv file to save indices results for scoring')
    parser.add_argument('--scoring_svr_rbf_model_file', dest='scoring_svr_rbf_model_file', action='store', 
                        default='svr_rbf_all.p', help='Pickle file (.p) where to save sklearn svr model')

    """parameters for fluctuations"""
    parser.add_argument('--fluctuations', dest='fluctuations', action='store_true', default=False)
    parser.add_argument('--fluctuations_lang', dest='fluctuations_lang', default=Lang.RO.value, nargs='?', 
                        choices=[Lang.RO.value, Lang.EN.value], help='Language for fluctuations (only ro and en supported)')

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