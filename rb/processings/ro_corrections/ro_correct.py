#!/usr/bin/env python
# coding: utf-8

import re
import json
import nltk
import spacy
import sys
from rb.parser.spacy_parser import SpacyParser
from rb.parser.spacy_parser import convertToPenn
from rb.core.lang import Lang
from nltk.stem.snowball import SnowballStemmer
from rb.processings.ro_corrections.get_pos_properties import *
from rb.processings.ro_corrections.get_exceptions import *
from rb.processings.ro_corrections.utils import *
from rb.utils.rblogger import Logger
import os

from nltk.stem.snowball import SnowballStemmer

logger = Logger.get_logger()

stemmer = SnowballStemmer("romanian")
rom_spacy = SpacyParser.get_instance().get_model(Lang.RO)

window_size = 8


with open(os.path.join(os.path.dirname(__file__), './lexicons.json'), 'rt') as fin:
    lexicons = json.load(fin)

with open(os.path.join(os.path.dirname(__file__), './exceptions.json'), 'rt') as fin:
    exceptions = json.load(fin)

def remove_punctuation(doc):
    punctuation_list = [".", ",", "!", ":", "?", ";"]
    result = []
    
    for el in doc:
        # if el.text not in punctuation_list:
        result.append(el)
    return result

def convert_char_ind_to_word_ind(text, index):
    if index == 0:
        return {'paragraph': 0, 'index': 0}
    if '\n' not in text[:index]:
        prev_endl_index = 0
    else:
        prev_endl_index = index - list(reversed(text[:index])).index('\n') - 1
    if prev_endl_index == 0:
        paragraph_no = 0
    elif '\n' not in text[:prev_endl_index]:
        paragraph_no = 1
    else:
        paragraph_no = len(text[:prev_endl_index].split('\n'))

    index = index - prev_endl_index - 1
    paragraph = text[(prev_endl_index + 1):].split('\n')[0]
    return {'paragraph': paragraph_no, 'index': len(rom_spacy(paragraph[:index]))}

def is_exception(mistake, case):
    mistake = " ".join(mistake)
    for exception in exceptions[case]:
        if mistake == exception.lower():
            return True
    return False

#for cacophonies and "ca_si" returns a list with (mistake, index)
def build_response(mistakes_list, sentence, case):
    items = []
    if case == "Cacofonie" or case == "Si_parazitar":
        item  = {
            'mistake': case,
            'index': mistake_list
        }
        items.append(item)
        return items
    if case == "Conjunctie coordonatoare" or case == "Verbul 'a voi'" or case == "Totusi" or case == "Adverbe la inceput" or case == "Adverbe intercalate":
        for mistake in mistakes_list:
            for match in re.finditer(mistake, sentence):
                index = match.start()
                test_index = index
                if case == 'Adverbe intercalate':
                    test_index += 2
                par_word = convert_char_ind_to_word_ind(sentence, test_index)
                paragraph_index = par_word['paragraph']
                word_index = par_word['index']
                
                item  = {
                    'mistake': mistake,
                    'paragraph': paragraph_index,
                    'word_index': [word_index], 
                    'start': match.start(),
                    'end': match.end()
                }
                if item not in items:
                    items.append(item)
        return items
    if case == "Conjunctie adversativa":
        for mistake in mistakes_list:
            for match in re.finditer(mistake, sentence):
                index = match.start() + 1
                par_word = convert_char_ind_to_word_ind(sentence, index)
                paragraph_index = par_word['paragraph']
                word_index = par_word['index']
                
                item  = {
                    'mistake': mistake,
                    'paragraph': paragraph_index,
                    'word_index': [word_index], 
                    'start': match.start() + 1,
                    'end': match.end()
                }
                if item not in items:
                    items.append(item)
        return items
    if case == "Comparativ de superioritate":
        for mistake in mistakes_list:
            for match in re.finditer(mistake, sentence):
                index = match.end() - 2
                par_word = convert_char_ind_to_word_ind(sentence, index)
                paragraph_index = par_word['paragraph']
                word_index = par_word['index']
                
                item  = {
                    'mistake': mistake,
                    'paragraph': paragraph_index,
                    'word_index': [word_index], 
                    'start': match.end() - 2, 
                    'end': match.end()
                }
                if item not in items:
                    items.append(item)
        return items 

def check_cacophony(par_index, doc):
    cac_list = [['la', 'la'], ['sa', 'sa'],
            ['ca', 'ca'], ['ca', 'că'],
            ['ca', 'ce'], ['ca', 'ci'],
            ['că', 'ca'], ['cu', 'co'],
            ['că', 'cu'], ['că', 'când'],
            ['ca', 'când'], ['că', 'câ']]
    cacophony_list = []
    for i, token in enumerate(doc):
        if i == len(doc) - 2:   continue
        for c_pair in cac_list:
            if doc[i].text == c_pair[0] and doc[i + 1].text == c_pair[1]:
                cacophony_list.append({
                    'mistake': "Cacofonie",
                    'index': [ [par_index, i], [par_index, i + 1] ]
                })
    return cacophony_list

# def check_si_parazitar(sentence):
#     regex = '(\w*ca)\s(şi\w*)'
#     p = re.compile(regex, re.IGNORECASE)
#     return build_response(p.findall(sentence), sentence, "Si_parazitar")

#group repetition in dictionary to be solve separated
def build_repetitions_dictionary(repetitions):
    d = {}
    for item in repetitions:
        if item[2] not in d:
            d[item[2]] = []
        d[item[2]].append((item[0], item[1]))
    return d


#using a window check if there are words with the same stem in window. Eliminate stop_words
#create a list with (word, index, stem)
#returns a dictionary with grouped repetitions
def check_repetitions(par_index, doc):
    
    repetitions = []
    index = 0
    
    while index == 0 or len(doc) - index >= window_size:
        repetitions_dictionary = {}
        #lemmatization
        for i in range(index, min(index + window_size, len(doc))):
            word = doc[i]
            num_endlines = len(list(filter(lambda x: len(x.text.strip()) == 0, doc[:i])))
            if word.is_stop != True:
                lemma = word.lemma_
                if lemma in repetitions_dictionary:
                    repetitions_dictionary[lemma].append(i)
                else:
                    repetitions_dictionary[lemma] = [i]
        for k, v in repetitions_dictionary.items():
            if len(v) >= 2 and doc[v[0]].is_punct == False and doc[v[0]].is_stop == False:
                el = [ [par_index, value] for value in v]
                if el not in repetitions:
                    repetitions.append(el)
        index += 1
    items = []
    for rep in repetitions:
        items.append({'mistake': "Repetiție",
                    'index': rep})
    return items


def are_synonyms(word1, word2):
    return ((len(set(lexicons[word1]) & set([word2])) >= 1) or (len(set(lexicons[word2]) & set([word1])) >= 1))

def get_synonym_pairs(window):
    pairs = []
    
    for i in range(len(window)):
        word_i = window[i].lemma_
        if window[i].is_stop or word_i not in lexicons:
            continue
        for j in range(i + 1, len(window)):
            word_j = window[j].lemma_
            if window[j].is_stop or word_j not in lexicons:
                continue
            if word_i != word_j and are_synonyms(word_i, word_j):
                pairs.append((i, j))
    
    return pairs

def get_synonym_clusters(synonyms):
    clusters = {}
    cluster_mappings = {}
    for i, j in synonyms:
        if i in clusters:
            clusters[i].append(j)
            cluster_mappings[j] = i
        elif i in cluster_mappings:
            if j not in clusters[cluster_mappings[i]]:
                clusters[cluster_mappings[i]].append(j)
                cluster_mappings[j] = cluster_mappings[i]
        else:
            clusters[i] = [i, j]
            cluster_mappings[i] = i
            cluster_mappings[j] = i
    return clusters

#check if a synonym is used in window
def check_synonyms_repetitions(sentence):

    doc = rom_spacy.parser(rom_spacy(sentence))
    sentence_tokens = remove_punctuation(list(doc))

    repetitions = []
    check = set()
    index = 0
    while index == 0 or len(sentence_tokens) - index >= window_size:
        repetitions_dictionary = {}
        window = [e for e in sentence_tokens[index: index + window_size] if len(e.text.strip()) != 0]
        num_endlines = len([e for e in sentence_tokens[index: index + window_size] if len(e.text.strip()) == 0])
        synonym_pairs = get_synonym_pairs(window)
        clusters = get_synonym_clusters(synonym_pairs)
        for cluster in clusters:
            common_key = window[clusters[cluster][0]].text
            for word_index in clusters[cluster]:
                actual_index = index + word_index
                item = (window[word_index].text, actual_index + 1 + num_endlines, common_key)
                if item not in check:
                    check.add(item)
                    repetitions.append((window[word_index].text, actual_index + 1, common_key))

        index += 1
    repetitions = build_repetitions_dictionary(repetitions)
    return convert_repetitions_dictionary(sentence, repetitions)

#Dupa totusi nu se pune virgula.
def check_totusi(par_index, doc):
    mistake_list = []
    for i, token in enumerate(doc):
        if i == 0:   continue
        if doc[i].text.lower() == 'totuși' and doc[i - 1].text == ',':
            mistake_list.append({
                'mistake': "Virgulă înainte de totuși",
                'index': [ [par_index, i - 1], [par_index, i] ]
            })
    return mistake_list

#Atunci când părțile de propoziție sau propozițiile se află într-un raport de coordonare, introduse prin conjuncțiile și, sau, ori, totuși nu se despart prin virgulă.
def check_coordinative_conjunctions(sentence):
    coordinative_conjunctions_list = ['(,\s*și)','[^\w](și\s*,)', '(,\s*sau)','(sau\s*,)', '(,\s*ori)','(ori\s*,)']
    mistakes = []
    for regex in coordinative_conjunctions_list:
        p = re.compile(regex, re.IGNORECASE)
        mistakes += p.findall(sentence)
    return build_response(mistakes, sentence, "Conjunctie coordonatoare")

def check_adversative_conjunctions(sentence):
    #nici, dar, iar, însă, ci, ci și, dar și, precum și, deci, prin urmare, așadar, în concluzie.
    adversative_conjunctions = ['[a-z]\s+dar\s', '[a-z]\s+nici\s', '[a-z]\s+iar\s', '[a-z]\s+însă\s', '[a-z]\s+ci\s','[a-z]\s+ci\s+și\s',
                               '[a-z]\s+dar\s+și\s]', '[a-z]\s+precum\s+și\s', '[a-z]\s+deci\s', '[a-z]\s+prin\s+urmare\s',
                               '[a-z]\s+așadar\s', '[a-z]\s+în\s+concluzie\s']
    mistakes = []
    for regex in adversative_conjunctions:
        p = re.compile(regex, re.IGNORECASE)
        mistakes += p.findall(sentence)
    return build_response(mistakes, sentence, "Conjunctie adversativa")


def check_voi_verb(par_index, doc):
    verb_list = ['vroiam', 'vroiai', 'vroia', 'vroiau', 'vroiaţi']
    mistake_list = []
    for i, token in enumerate(doc):
        if i == len(doc) - 2:   continue
        for vb in verb_list:
            if doc[i].text == vb:
                mistake_list.append({
                    'mistake': "Forme verbului",
                    'index': [ [par_index, i] ]
                })
    return mistake_list

def check_comparative(sentence):
    regex_list = ['(mai\s\w+\sde)', '(mai\s\w+\sca)']
    mistakes = []
    for regex in regex_list:
        p = re.compile(regex, re.IGNORECASE)
        mistakes += p.findall(sentence)
    return build_response(mistakes, sentence, "Comparativ de superioritate")

def check_adverbs_at_the_beginning(par_index, doc):

    adverb_list = ['Totuși', 'Așadar', 'Prin urmare', 'Deci']
    mistake_list = []

    for i, token in enumerate(doc):
        if i == len(doc) - 2:   continue
        for vb in adverb_list:
            if doc[i].text == vb:
                mistake_list.append({
                    'mistake': "Adevrb nepotrivit la început de frază",
                    'index': [ [par_index, i] ]
                })
    return mistake_list

def check_adverbs_in_middle(sentence):
    adverb_list = ['(\w\s+desigur)','(desigur\s+\w)', '(\w\s+firește)', '(firește\s+\w)','(\w\s+așadar)','(așadar\s+\w)',
                   '(\w\s+bineînțeles)', '(bineînțeles\s+\w)','(\w\s+în concluzie)', '(în concluzie\s+\w)', 
                   '(\w\s+în realitate)', '(în realitate\s+\w)', '(\w\s+de exemplu)', '(de exemplu\s+\w)']
    mistakes = []
    for regex in adverb_list:
        p = re.compile(regex)
        mistakes += p.findall(sentence)
    return build_response(mistakes, sentence, "Adverbe intercalate")

def check_verbal_predicate(par_index, S, P, spn, corrections):
    try:
        if P.tag_[:3] == "Vmi":
            spv = get_verb(P.tag_)
            if is_collective_subject(S, spv, exceptions):
                return None

            if spn[:2] == spv:
                return None
            if spn[0] == spv[0] and spv[1] == "":
                return None
            if spn[1] == spv[1] and spv[0] == "-":
                return None
            
            person, number, gender = get_person_number_gender(spn)
            mistake = {
                'mistake': 'Acordul verbului cu subiectul',
                'index': [[par_index, S.i], [par_index, P.i]]
            }
            corrections.append(mistake)
            
        elif P.tag_[:3] == "Vmp" or P.tag_[:4] == "Vmnp":
            for child in P.children:
                if child.dep_ == "aux":
                    if child.tag_ == "Vanp":
                        return None

                    aux = get_verb(child.tag_)
                    if is_collective_subject(S, aux, exceptions):
                        return None

                    if aux == spn[:2]:
                        return None
                    if aux[0] == spn[0] and aux[1] == "":
                        return None

                    mistake = {
                        'mistake': 'Acordul verbului cu subiectul',
                        'index': [[par_index, S.i], [par_index, child.i]]
                    }
                    corrections.append(mistake)
    except:
        return None

def check_copulative_verb(par_index, S, P, spn, corrections):
    #check copulativ verb with subject
    has_cop = False
    for child in P.children:
        if child.dep_ == "cop":
            if child.tag_[:3] == "Vmi" or child.tag_[:3] == "Vai":
                spv = get_verb(child.tag_)

                if is_collective_subject(S, spv, exceptions):
                    return None
                
                if spn[:2] == spv:
                    return None
                if spn[0] == spv[0] and spv[1] == "":
                    return None
                if spn[1] == spv[1] and spv[0] == "-":
                    return None
                # predicat nominal: verb copulativ + nume predicativ, punele pe ambele in index
                mistake = {
                    'mistake': 'Acordul verbului cu subiectul',
                    'index': [[par_index, S.i], [par_index, P.i], [par_index, child.i]]
                }
                corrections.append(mistake)
            else:
                has_cop = True
    if has_cop == True:
        for child1 in P.children:
            if child1.dep_ == "aux":
                if child1.tag_ == "Vanp":
                    return None
                aux = get_verb(child1.tag_)
                if is_collective_subject(S, aux, exceptions):
                    return None

                if aux == spn[:2]:
                    return None
                if aux[0] == spn[0] and aux[1] == "":
                    return None

                mistake = {
                    'mistake': 'Acordul verbului cu subiectul',
                    'index': [[par_index, S.i], [par_index, child.i]]
                }
                corrections.append(mistake)

def check_predicative_name(par_index, S, NP, spn, corrections):
    if is_job_noun(spn, NP, exceptions):
        return None

    if NP.tag_[:3] == "Rgp":
        return None

    np = ""
    #adjectiv
    if NP.tag_[:2] == "Af":
        np = get_adjective(NP.tag_)
    #noun
    elif NP.tag_[:2] == "Nc":
       np = get_noun(NP.tag_)

    #ordinal numeral or cardinal numeral
    elif NP.tag_[:2] == "Mo" or NP.tag_[:2] == "Mc":
       np = get_numeral(NP.tag_)

    if np[1] == spn[1] and np[2] == spn[2]:
       return None
    #Subject is a personal pronoun without gender
    if np[1] == spn[1] and spn[2] == "-":
       return None

    mistake = {
        'mistake': 'Acordul numelui predicativ cu subiectul',
        'index': [[par_index, S.i], [par_index, NP.i]]
    }
    corrections.append(mistake)


def check_subject_and_predicate_relation(par_index, doc):
    corrections = []
    predicates = []

    for token in doc:
        if token.dep_ == "nsubj":
            S = token
            P = token.head
            if P not in predicates:
                predicates.append(P)
            else:
                continue

            if is_multiple_subject_linked_to_subject(S):
                spn = get_multiple_subject(S, "conj", exceptions)
            elif is_multiple_subject_linked_to_predicate(P):
                spn = get_multiple_subject(P, "nsubj", exceptions)
            else:
                if S.tag_[0] == "P":
                    spn = get_pronoun(S.tag_)
                else:
                    spn = get_noun(S.tag_)

            multiple_predicates = get_multiple_predicates(P)

            #all parts from multiple_predicates shoud be in the same relation with the subject
            for single_predicate in multiple_predicates:
                if single_predicate.tag_[0] == "V":
                    check_verbal_predicate(par_index, S, single_predicate, spn, corrections)
                else:
                    check_copulative_verb(par_index, S, single_predicate, spn, corrections)
                    check_predicative_name(par_index, S, single_predicate, spn, corrections)
    return corrections

def check_noun_and_adjective_relation(par_index, doc):
    corrections = []
    for token in doc:
        if token.tag_[:2] == "Af":
            (is_linked, noun) = is_in_adjective_noun_relation(token)
            if is_linked:
                A = token
                N = noun
                spa = get_adjective(A.tag_)
                spn = get_noun(N.tag_)
                if spa[1] == spn[1] and spa[2] == spn[2]:
                    continue
                else:
                    mistake = {
                        'mistake': 'Acord substantiv - adjectiv',
                        'index': [[par_index, A.i], [par_index, N.i]]
                    }
                    corrections.append(mistake)
    return corrections

def check_noun_and_numeral_relation(par_index, doc):
    corrections = []
    for token in doc:
        if token.dep_ == "nummod":
            if token.tag_[:2] == "Mc" or token.tag_[:2] == "Mo":
                Nr = token
                N = token.head
                spnr = get_numeral(Nr.tag_)
                spn = get_noun(N.tag_)
                if spnr[1] == spn[1] and spnr[2] == spn[2]:
                    continue
                if spnr[2] == "-" and spnr[1] == spn[1]:
                    continue
                else:
                    #it is not correct
                    mistake = {
                        'mistake': 'Acord substantiv - numeral',
                        'index': [[par_index, Nr.i], [par_index, N.i]]
                    }
                    corrections.append(mistake)
    return corrections

def check_noun_and_unstated_article_relation(par_index, doc):
    corrections = []
    for token in doc:
        if token.dep_ == "det":
            if token.tag_[:2] == "Ti":
                UA = token
                N = token.head
                spar = get_unstated_article(UA.tag_)
                spn = get_noun(N.tag_)
                if spar[1] == spn[1] and spar[2] == spn[2]:
                    continue
                if spar[2] == "-" and spar[1] == spn[1]:
                    continue
                else:
                    #it is not correct

                    mistake = {
                        'mistake': "Acord substantiv - acord nehotarat",
                        'index': [[par_index, token.i], [par_index, N.i]]
                    }
                    corrections.append(mistake)
            elif token.tag_[:3] == "Di3":
                UA = token.head
                N = token.head
                spn = get_noun(N.tag_)
                if spn[2] == "p":
                    continue
                else:
                    #it is not correct
                    mistake = {
                        'mistake': "Acord substantiv - acord nehotarat",
                        'index': [[par_index, token.i], [par_index, N.i]]
                    }
                    corrections.append(mistake)
    return corrections


def create_output(category, message, paragraph_index, word_index):
    output = {}
    output["title"] = category
    output["message"] = message
    output["paragraph_index"] = paragraph_index
    output["word_index"] = word_index
    return output

def get_mistakes(mistakes):
    message = []
    for item in mistakes:
        message.append("(" + str(item[0]) + ")")
    return ", ".join(message)

def create_output_list(result, case, default_suggestion = None):
    output_list = []
    if case == "Cacofonie":
        for item in result:
            message = "Reformulare. Cacofonie."
            output_list.append(create_output(case, message, paragraph_index, item["word_index"][0]))
            output_list.append(create_output(case, message, paragraph_index, item["word_index"][1]))
        return output_list
    if case == "şi parazitar":
        for item in result: 
            message = "Expresia (" + item["mistake"] + ") Şi parazitar (apariția lui nu este permisă nici pentru evitarea cacofoniei). Reformulare folosind una dintre expresiile: 'în calitate de' sau 'drept'"
            paragraph_index = item["paragraph"]
            output_list.append(create_output(case, message, paragraph_index, item["word_index"][0]))
            output_list.append(create_output(case, message, paragraph_index, item["word_index"][1]))
        return output_list
    if case == "Repetiţie" or case == "Repetiţie sinonime":
        for mistake_list in result:
            mistake_output_list = []
            for item in mistake_list:
                try:
                    if len(item['mistake']) >= 1 and (str(item['mistake'][0][0]) == '.' or str(item['mistake'][0][0]) == ','):
                        continue
                except:
                    pass
                message = "Reformulare. Repetiţie deranjantă pentru cuvântul/cuvintele: " + get_mistakes(item["mistake"])
                if case == "Repetitie sinonime":
                    message += " Cuvintele sunt sinonime."
                paragraph_index = item["paragraph"]
                word_index = item["word_index"]
                mistake_output_list.append(create_output(case, message, paragraph_index, word_index))
                # print('len mistake output list', len(mistake_output_list))
                # print(mistake_output_list)
            output_list.append(mistake_output_list)
        return output_list
    if case == "Conjuncţie coordonatoare":
        for item in result:
            message = "Atunci când părțile de propoziție sau propozițiile se află într-un raport de coordonare, introduse prin conjuncțiile și, sau, ori, nu se despart prin virgulă. Varianta corectă: " + default_suggestion
            paragraph_index = item["paragraph"]
            word_index = item["word_index"]
            output_list.append(create_output(case, message, paragraph_index, word_index))
        return output_list
    if case == "Conjuncţie adversativă":
        for item in result:
            message = "Virgula este obligatorie în fața conjuncțiilor/ locuțiunilor coordonatoare adversative. Varianta corectă: " + default_suggestion
            paragraph_index = item["paragraph"]
            word_index = item["word_index"]
            output_list.append(create_output(case, message, paragraph_index, word_index))
        return output_list
    if case == "Verbul 'a voi'":
        for item in result:
            message = "Forma corectă de imperfect a verbului 'a voi' este 'voiam' la persoana I etc. Varianta corectă: " + default_suggestion
            paragraph_index = item["paragraph"]
            word_index = item["word_index"]
            output_list.append(create_output(case, message, paragraph_index, word_index))
        return output_list
    if case == "Comparativ de superioritate":
        for item in result:
            message = "Comparativul de superioritate se exprimă cu ajutorul lui „decât”. Varianta corectă: " + default_suggestion
            paragraph_index = item["paragraph"]
            word_index = item["word_index"]
            output_list.append(create_output(case, message, paragraph_index, word_index))
        return output_list
    if case == "Totuşi":
        for item in result:
            message = "După 'totuși' nu se pune virgulă, dacă acesta este în interiorul propoziției. Varianta corectă: " + default_suggestion
            paragraph_index = item["paragraph"]
            word_index = item["word_index"]
            output_list.append(create_output(case, message, paragraph_index, word_index))
        return output_list
    if case == "Adverbe la începutul propoziţiei":
        for item in result:
            message = "La începutul propoziției, se pune virgulă după termenii: 'totuşi', 'aşadar', 'prin urmare' şi 'deci'. Varianta corectă: " + default_suggestion
            paragraph_index = item["paragraph"]
            word_index = item["word_index"]
            output_list.append(create_output(case, message, paragraph_index, word_index))
        return output_list
    if case == "Adverbe intercalate":
        for item in result:
            message = "Intercalarea de adverbe, precum 'desigur', 'fireşte', 'aşadar', 'bineînţeles', 'în concluzie', 'în realitate', 'de exemplu' impune punerea virgulei. Varianta corectă: " + default_suggestion
            paragraph_index = item["paragraph"]
            word_index = item["word_index"]
            output_list.append(create_output(case, message, paragraph_index, word_index))
        return output_list


def sort_by_start_index(result):
    return sorted(result, key = lambda x: x['start'])


def corrects_coordonative_error(result, sentence):
    result = sort_by_start_index(result)
    
    sentence = list(sentence)
    for item in result:
        start = item["start"]
        end = item["end"]
        if sentence[start] == ",":
            sentence[start] = " "
        if sentence[end - 1] == ",":
            sentence[end - 1] = " "
    sentence = "".join(sentence)
    p = re.compile('\s\s+')
    return p.sub(' ', sentence)


def corrects_adversative_error(result, sentence):
    result = sort_by_start_index(result)
    
    sentence = list(sentence)
    for item in result:
        start = item["start"]
        sentence[start] = ", "
    sentence = "".join(sentence)
    p = re.compile('\s\s+')
    return p.sub(' ', sentence)


def corrects_voi_verb(result, sentence):
    result = sort_by_start_index(result)
    
    sentence = list(sentence)
    index = 0
    for item in result:
        del sentence[item["start"] + 1 - index]
        index += 1
    return "".join(sentence)


def corrects_comparative(result, sentence):
    result = sort_by_start_index(result)
    
    index = 0
    for item in result:
        sentence = sentence[0:item["start"] + index * 3] + "decât" + sentence[item["end"] + index * 3:]
        index += 1
    return sentence

def corrects_adverbs_at_the_beginning(result, sentence):
    result = sort_by_start_index(result)
    
    sentence = list(sentence)
    for item in result:
        end = item["end"]
        sentence[end - 1] = ", "
    sentence = "".join(sentence)
    p = re.compile('\s\s+')
    return p.sub(' ', sentence)



def corrects_adverbs_in_middle(result, sentence):
    result = sort_by_start_index(result)
    
    p = re.compile('\s\s+')
    sentence = list(sentence)
    for item in result:
        start = item["start"]
        end = item["end"]
        if sentence[start + 1] == " ":
            sentence[start + 1] = ", "
        if sentence[end - 2] == " ":
            sentence[end - 2] = ", "
    
    sentence = "".join(sentence)
    return p.sub(' ', sentence)
            


def split_text(sentence):
    split_text = []
    for sentence in sentence.split('\n'):
        words = [token.text for token in rom_spacy(sentence)]
        split_text.append([words])
    return split_text

def get_paragraph_offset(pid, text):
    offset = 0
    for par in text[:pid]:
        for word in par[0]:
            offset += len(word)
        offset += len(par[0]) - 1
    return offset

def get_word_offset(wid, paragraph, end = False):
    offset = 0
    for word in paragraph[0][:wid]:
        offset += len(word) + 1
        if word == '-':
            offset -= 2
    if end:
        offset += len(paragraph[0][wid]) - 1
    return offset


def fix_dashes(output):
    word_breaks = []
    new_text = []
    for paragraph in output['split_text']:
        par_breaks = []
        new_paragraph = [[]]
        word_breaks.append(par_breaks)
        new_text.append(new_paragraph)

        i = 0
        while i < len(paragraph[0]):
            if paragraph[0][i] == '-':
                par_breaks.append(i)
                new_paragraph[0][-1] += paragraph[0][i] + paragraph[0][i + 1]
                i += 2
            else:
                new_paragraph[0].append(paragraph[0][i])
                i += 1
    
    output['split_text'] = new_text
    for error_l in output['correction']:
        if isinstance(error_l, list):
            for error in error_l:
                paragraph = error['paragraph_index']
                breaks = word_breaks[paragraph]
                if len(breaks) == 0:
                    continue
                
                count = 0
                for i in range(len(error['word_index'])):
                    offset = 0
                    for e in breaks:
                        if error['word_index'][i] > e:
                            offset -= 2
                    error['word_index'][i] += offset
        else:
            error = error_l
            paragraph = error['paragraph_index']
            breaks = word_breaks[paragraph]
            if len(breaks) == 0:
                continue
            
            count = 0
            for i in range(len(error['word_index'])):
                offset = 0
                for e in breaks:
                    if error['word_index'][i] > e:
                        offset -= 2
                error['word_index'][i] += offset


def fix_punctuation(output):
    import string
    word_breaks = []
    new_text = []
    for paragraph in output['split_text']:
        par_breaks = []
        new_paragraph = [[]]
        word_breaks.append(par_breaks)
        new_text.append(new_paragraph)

        i = 0
        while i < len(paragraph[0]):
            if paragraph[0][i] != '-' and paragraph[0][i] in ".,;:!?":
                par_breaks.append(i)
                new_paragraph[0][-1] += paragraph[0][i]
                i += 1
            else:
                new_paragraph[0].append(paragraph[0][i])
                i += 1
    
    output['split_text'] = new_text
    for error_l in output['correction']:
        if isinstance(error_l, list):
            for error in error_l:
                paragraph = error['paragraph_index']
                breaks = word_breaks[paragraph]
                if len(breaks) == 0:
                    continue
                
                count = 0
                for i in range(len(error['word_index'])):
                    offset = 0
                    for e in breaks:
                        if error['word_index'][i] > e:
                            offset -= 1
                    error['word_index'][i] += offset
        else:
            error = error_l
            paragraph = error['paragraph_index']
            breaks = word_breaks[paragraph]
            if len(breaks) == 0:
                continue
            
            count = 0
            for i in range(len(error['word_index'])):
                offset = 0
                for e in breaks:
                    if error['word_index'][i] > e:
                        offset -= 1
                error['word_index'][i] += offset


def change_format(par_index, output):
    for error_l in output["correction"]:
        if isinstance(error_l, list):
            for error in error_l:
                if isinstance(error["word_index"], int):
                    error["word_index"] = [error["word_index"]]
        else:
            error = error_l
            if isinstance(error["word_index"], int):
                error["word_index"] = [error["word_index"]]
    
    fix_dashes(output)
    fix_punctuation(output)
    mistakes = []
    for error_l in output["correction"]:
        if isinstance(error_l, list):
            for error in error_l:
                par_offset = get_paragraph_offset(error["paragraph_index"], output["split_text"])
                word_offset_start = get_word_offset(error["word_index"][0], output["split_text"][error["paragraph_index"]])
                word_offset_end = get_word_offset(error["word_index"][-1], output["split_text"][error["paragraph_index"]], True)
                error["correction_index"] = [par_offset + word_offset_start, par_offset + word_offset_end]
        else:
            error = error_l
            par_offset = get_paragraph_offset(error["paragraph_index"], output["split_text"])
            word_offset_start = get_word_offset(error["word_index"][0], output["split_text"][error["paragraph_index"]])
            word_offset_end = get_word_offset(error["word_index"][-1], output["split_text"][error["paragraph_index"]], True)
            error["correction_index"] = [par_offset + word_offset_start, par_offset + word_offset_end]
    return output


def simplify_format(output):
    
    new_corrections = []

    for error_l in output['correction']:
        if isinstance(error_l, list):
            if len(error_l) > 0:
                refact_cor = {}
                correction_index_list = []
                refact_cor['message'] = error_l[0]['message']
                refact_cor['title'] = error_l[0]['title']
                for error in error_l:
                    correction_index_list.append(error['correction_index'])
                refact_cor['correction_index'] = correction_index_list
                new_corrections.append(refact_cor)
        else:
            error_l['correction_index'] = [error_l['correction_index']]
            if 'paragraph_index' in error_l:
                del error_l['paragraph_index']
            if 'word_index' in error_l:
                del error_l['word_index']
            new_corrections.append(error_l)
    res = {}
    res['corrections'] = new_corrections
    res['split_text'] = output['split_text']
    return res

def identify_mistake(par_index, sentence, spellchecking=False):

    doc = rom_spacy(sentence)
    output_list = []

    if spellchecking == True:
        spellcheck_list = []
        import hunspell as hs
        ro_hunspell = hs.HunSpell('/usr/share/hunspell/ro_RO.dic', '/usr/share/hunspell/ro_RO.aff')
        for i, token in enumerate(doc):
            if token.is_punct == False and ro_hunspell.spell(token.text) == False:
                spellcheck_list.append({
                    'mistake': "Ortografie",
                    'index': [ [par_index, i], [par_index, i + 1] ],
                    "suggestions": ro_hunspell.suggest(token.text)
                })

    output_list += spellcheck_list
    # print(dir(doc[0]))
    # for i, token in enumerate(doc):
    #     print('index', token.text, token.pos_, token.is_stop, token.is_punct, token.i)

    correct = True
    #Step1
    try:
        result = check_cacophony(par_index, doc)
        output_list += result
    except:
        logger.warning('Cachophony check failed')
        
    
    #Step2 - ca si
    # result = check_si_parazitar(par_index, sentence) - does not work all the time
    # output_list += result
        
    #Step3 - Repetitions
    try:
        result = check_repetitions(par_index, doc)
        output_list += result
    except:
        logger.warning('Repetitions check failed')
        
    #Step4 - Conjunctii coordonatoare - does not work all the time
    # result = check_coordinative_conjunctions(sentence) 
    # if result != []:
    #     correct = False
    #     output_list += create_output_list(result, "Conjuncţie coordonatoare", corrects_coordonative_error(result, sentence))

    #Step5 - Conjunctii adversative
    # result = check_adversative_conjunctions(sentence)
    # if result != []:
    #     correct = False
    #     output_list += create_output_list(result, "Conjuncţie adversativă", corrects_adversative_error(result, sentence))

    #Step6 - Repetitie sinonime
    # result = check_synonyms_repetitions(sentence.lower())
    # if result != {}:
    #     correct = False
    #     output_list += create_output_list(result, "Repetiţie sinonime")
    
    #Step7 - Voi
    try:
        result = check_voi_verb(par_index, doc)
        output_list += result
    except:
        logger.warning('Voi verb check failed')

        
    #Step8 - Comparativ de superioritate - does not work all the time
    # result = check_comparative(sentence)
    # if result != []:
    #     correct = False
    #     output_list += create_output_list(result, "Comparativ de superioritate", corrects_comparative(result, sentence))

    #Step9 - Totusi
    try:
        result = check_totusi(par_index, doc)
        output_list += result
    except:
        logger.warning('Totusi check failed')

    try:
        output_list += check_adverbs_at_the_beginning(par_index, doc) 
    except:
         logger.warning('Adverbs at the beginning check failed')
    #Step11 - Adverbe intercalate
    # result = check_adverbs_in_middle(sentence)
    # if result != []:
    #     correct = False
    #     output_list += create_output_list(result, "Adverbe intercalate", corrects_adverbs_in_middle(result, sentence))

    #Step12 - Subject and predicate
    try:
        output_list += check_subject_and_predicate_relation(par_index, doc)
    except:
        logger.warning('Subject and predicate relation check failed')
    # #Step13 - Noun and adjective relation
    try:
        output_list += check_noun_and_adjective_relation(par_index, doc)
    except:
        logger.warning('Noun and adjective relation check failed')

    # #Step14 - Noun and numeral relation
    try:
        output_list += check_noun_and_numeral_relation(par_index, doc)
    except:
        logger.warning('Noun and numeral relation check failed')
    # #Step15 - Noun and unstated article relation
    try:
        output_list += check_noun_and_unstated_article_relation(par_index, doc)
    except:
        logger.warning('Noun and unstated article relation check failed')
    return output_list, [token.text for token in doc]


def correct_text_ro(text, spellchecking=False):
    p_newline_after = re.compile('\n\s+')
    p_newline_before = re.compile('\s+\n')
    text = p_newline_after.sub('\n', text)
    text = p_newline_before.sub('\n', text)
    paragraphs = text.split('\n')
    paragraphs = [p for p in paragraphs if len(p) > 0]

    output = {'split_text': [], 'corrections': []} 
    for i, paragraph in enumerate(paragraphs):
        mistakes, text_splitted = identify_mistake(i, paragraph, spellchecking=spellchecking)
        output['split_text'].append(text_splitted)
        output['corrections'] += mistakes
    return output

if __name__ == "__main__":
    txt = "Fiind protejate de stratul de gheaţă, apele mai adânci nu îngheaţă până la fund, ci au, sub stratul de gheaţă, temperatura de 4 grade la care viaţa poate continua"
    correct_text_ro(txt)
    with open('log.log', 'wt', encoding='utf-8') as f:
        f.write(jsonify(correct_text_ro(txt)))