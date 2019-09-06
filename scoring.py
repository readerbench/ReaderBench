from rb.processings.essay_scoring import EssayScoring
from rb.complexity.complexity_index import compute_indices
from rb.core.lang import Lang
from rb.core.document import Document
import csv

log = open('log.log', 'wt', encoding='utf-8')
test = """I. Evaluarea performantelor profesionale

Motto:

"Daca nu poti masura ceva, nu-l poti intelege,
Daca nu-l poti intelege, nu-l poti controla,
Daca nu-l poti controla, nu-l poti imbunatati."

H.I.Harrington

Capacitatea de cunoastere a omului este inseparabila de abilitatea sa de a evalua, care face posibila selectia, ierarhizarea si sistematizarea informatiilor. Fie ca vizeaza obiective, fenomene sau procese, plasarea pe o anumita pozitie a unei scari valorice determina in mod curent atitudini, decizii si actiuni.

Din perspectiva manageriala, aprecierea rezultatelor unei activitati, raportate la obiectivele organizatiei si in relatie cu contextul real in care se defasoara, constituie o conditie a oricarui demers de perfectionare sau de adaptare. La nivel individual, de organizatie sau sistem, evaluarea corect efectuata permite intelegerea clara a deficientelor si deschide calea unor posibile imbunatatiri.1

I.1. Tipuri de evaluare

Evaluarea ca notiune are un caracter complex, determinat de diversi factori functie de care se realizeaza. Astfel, unii autori2 considera evaluarea personalului ca fiind"actul prin care un responsabil ierarhic efectueaza o apreciere formalizata a subordonatilor sai", iar daca aceste aprecieri au loc anual atunci este vorba despre"un sistem de apreciere" al organizatiei. Alti autori fac referire doar la performantele obtinute, fiind considerata"activitatea de baza a managementului resurselor umane, desfasurata in vederea determinarii gradului in care angajatii unei organizatii indeplinesc eficient sarcinile si responsabilitatile care le revin"3"""


if __name__ == "__main__":

    input_csv_file = 'essays.csv'
    output_csv_indices = 'essays_eval_all.csv'
    svr_model_file = 'svr_rbf_all.p'


    essay_scoring = EssayScoring()
    essay_scoring.compute_indices(csv_file_in=input_csv_file, lang=Lang.RO, write_file=output_csv_indices)
    results = essay_scoring.read_indices(path_to_csv_file=output_csv_indices)
    essay_scoring.train_svr(results, save_model_file=svr_model_file)
    essay_scoring.predict(test, file_to_svr_model=svr_model_file)
    
        
