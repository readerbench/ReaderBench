#!/usr/bin/env python
# coding: utf-8

import re
import json
import nltk
import spacy
#integrate ReaderBench
import sys
from rb.parser.spacy_parser import SpacyParser
from rb.parser.spacy_parser import convertToPenn
from rb.core.lang import Lang
from nltk.stem.snowball import SnowballStemmer
from rb.ro_corrections.get_pos_properties import *
from rb.ro_corrections.get_exceptions import *
from rb.ro_corrections.utils import *
import os

from nltk.stem.snowball import SnowballStemmer

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
        last_index = 0
        for mistake in mistakes_list:
            if (is_exception(mistake, "Cacofonie")) :
                continue
            concatenated_mistake = mistake[0] + " " + mistake[1]
            index = sentence[last_index:].index(concatenated_mistake)
            par_word = convert_char_ind_to_word_ind(sentence, last_index + index)
            
            paragraph_index = par_word['paragraph']
            word_index = par_word['index']
            
            item  = {
                'mistake': concatenated_mistake,
                'paragraph': paragraph_index,
                'word_index': [word_index, word_index + 1]
            }
            last_index = index + len(mistake) - 1
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

def check_cacophony(sentence):
    regex_list = ['(\w*la)\s(la\w*)','(\w*sa)\s(sa\w*)','(\w*ca)\s(ca\w*)','(\w*ca)\s(co\w*)','(\w*ca)\s(că\w*)','(\w*ca)\s(ce\w*)',
            '(\w*ca)\s(ci\w*)','(\w*că)\s(ca\w*)','(\w*cu)\s(co\w*)', '(\w*că)\s(co\w*)', '(\w*că)\s(cu\w*)', '(\w*că)\s(când\w*)', 
            '(\w*ca)\s(când\w*)', '(\w*că)\s(câ)']
    cacophony_list = []
    for regex in regex_list:
        p = re.compile(regex, re.IGNORECASE)
        cacophony_list += p.findall(sentence)
    cacophony_list.sort(key=lambda s: sentence.index(s[0] + ' ' + s[1]))
    return build_response(cacophony_list, sentence, "Cacofonie")

def check_si_parazitar(sentence):
    regex = '(\w*ca)\s(şi\w*)'
    p = re.compile(regex, re.IGNORECASE)
    return build_response(p.findall(sentence), sentence, "Si_parazitar")

#group repetition in dictionary to be solve separated
def build_repetitions_dictionary(repetitions):
    d = {}
    for item in repetitions:
        if item[2] not in d:
            d[item[2]] = []
        d[item[2]].append((item[0], item[1]))
    return d

def convert_repetitions_dictionary(sentence, repetitions_dictionary):
    items = []
    for root in repetitions_dictionary:
        
        for i in range(len(repetitions_dictionary[root])):
            word_index = repetitions_dictionary[root][i][1] - 1
            text_before = ' '.join([w.text for w in rom_spacy(sentence)[:word_index]])

            paragraph_index = text_before.count('\n')
            previous_words = len(
                [w.text
                for w in rom_spacy('\n'.join(text_before.split('\n')[:-1]))])
            items.append({
                'mistake': repetitions_dictionary[root],
                'paragraph': paragraph_index,
                'word_index': word_index - previous_words,
            })
    return items

#using a window check if there are words with the same stem in window. Eliminate stop_words
#create a list with (word, index, stem)
#returns a dictionary with grouped repetitions
def check_repetitions(sentence):
    
    doc = rom_spacy.parser(rom_spacy(sentence))
    sentence_tokens = remove_punctuation(list(doc))
    sentence_tokens = [e for e in sentence_tokens if e.text.strip() != '...']
    
    repetitions = []
    index = 0
    lemmas = []
    
    last_index = 0
    while index == 0 or len(sentence_tokens) - index >= window_size:
        repetitions_dictionary = {}
        repetitions_dictionary_stemming = {}
        
        #lemmatization
        for i in range(index, min(index + window_size, len(sentence_tokens))):
            word = sentence_tokens[i]
            num_endlines = len(list(filter(lambda x: len(x.text.strip()) == 0, sentence_tokens[:i])))
            if word.is_stop != True:
                lemma = word.lemma_
                if lemma in repetitions_dictionary:
                    added_word = repetitions_dictionary[lemma]
                    if added_word not in repetitions:
                        repetitions.append(added_word)
                        lemmas.append(added_word[0])
                    if (word.text, i + 1 - num_endlines, lemma) not in repetitions:
                        repetitions.append((word.text, i + 1 - num_endlines, lemma))
                        lemmas.append(word.text)
                else:
                    repetitions_dictionary[lemma] = (word.text, i + 1 - num_endlines, lemma)
                    
        # try also with stemming
        for i in range(index, min(index + window_size, len(sentence_tokens))):
            word = sentence_tokens[i]
            num_endlines = len(list(filter(lambda x: len(x.text.strip()) == 0, sentence_tokens[:i])))
            if word.is_stop != True and word.text not in lemmas:
                stem = stemmer.stem(word.text)
                if stem in repetitions_dictionary_stemming:
                    added_word = repetitions_dictionary_stemming[stem]
                    if added_word not in repetitions:
                        repetitions.append(added_word)
                    if (word.text, i + 1 - num_endlines, stem) not in repetitions:
                        repetitions.append((word.text, i + 1 - num_endlines, stem))
                else:
                    repetitions_dictionary_stemming[stem] = (word.text, i + 1 - num_endlines, stem)
                    
        index += 1

    repetitions_dictionary = build_repetitions_dictionary(repetitions)
    return convert_repetitions_dictionary(sentence, repetitions_dictionary)


def are_synonyms(word1, word2):
    intersection  = set(lexicons[word1]) & set(lexicons[word2])
    return len(intersection) > 1

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
def check_totusi(sentence):
    regex = '(totuși\s*,)'
    p = re.compile(regex)
    mistakes = p.findall(sentence)
    return build_response(mistakes, sentence, "Totusi")

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


def check_voi_verb(sentence):
    verb_list = ['vroiam', 'vroiai', 'vroia[^\w]', 'vroiau', 'vroiaţi']
    mistakes = []
    for regex in verb_list:
        p = re.compile(regex, re.IGNORECASE)
        mistakes += p.findall(sentence)
    return build_response(mistakes, sentence, "Verbul 'a voi'")

def check_comparative(sentence):
    regex_list = ['(mai\s\w+\sde)', '(mai\s\w+\sca)']
    mistakes = []
    for regex in regex_list:
        p = re.compile(regex, re.IGNORECASE)
        mistakes += p.findall(sentence)
    return build_response(mistakes, sentence, "Comparativ de superioritate")

def check_adverbs_at_the_beginning(sentence):
    adverb_list = ['Totuși\s', 'Așadar\s', 'Prin urmare\s', 'Deci\s']
    mistakes = []
    for regex in adverb_list:
        p = re.compile(regex)
        mistakes += p.findall(sentence)
    return build_response(mistakes, sentence, "Adverbe la inceput")

def check_adverbs_in_middle(sentence):
    adverb_list = ['(\w\s+desigur)','(desigur\s+\w)', '(\w\s+firește)', '(firește\s+\w)','(\w\s+așadar)','(așadar\s+\w)',
                   '(\w\s+bineînțeles)', '(bineînțeles\s+\w)','(\w\s+în concluzie)', '(în concluzie\s+\w)', 
                   '(\w\s+în realitate)', '(în realitate\s+\w)', '(\w\s+de exemplu)', '(de exemplu\s+\w)']
    mistakes = []
    for regex in adverb_list:
        p = re.compile(regex)
        mistakes += p.findall(sentence)
    return build_response(mistakes, sentence, "Adverbe intercalate")

def check_verbal_predicate(S, P, spn, corrections, last, text, paragraph_index):
    if P.tag_[:3] == "Vmi":
        spv = get_verb(P.tag_)
        if is_collective_subject(S, spv, exceptions):
            return last

        if spn[:2] == spv:
            return last
        if spn[0] == spv[0] and spv[1] == "":
            return last
        if spn[1] == spv[1] and spv[0] == "-":
            return last
        
        #it is not correct
        word_index = get_index(text, last, P.text)
        person, number, gender = get_person_number_gender(spn)
        message = "Predicatul trebuie sa fie la persoana " + person + ", numarul " + number
        corrections.append({"message": message, "paragraph_index": paragraph_index, "title": "Acord subiect - predicat", "word_index": word_index})
        last = word_index
        return last
    #participiu  or infinitive
    elif P.tag_[:3] == "Vmp" or P.tag_[:4] == "Vmnp":
        for child in P.children:
            if child.dep_ == "aux":
                if child.tag_ == "Vanp":
                    return last

                aux = get_verb(child.tag_)
                if is_collective_subject(S, aux, exceptions):
                    return last

                if aux == spn[:2]:
                    return last
                if aux[0] == spn[0] and aux[1] == "":
                    return last
                
                #it is not correct
                word_index = get_index(text, last, child.text)
                person, number, gender = get_person_number_gender(spn)
                message = "Predicatul trebuie sa fie la persoana " + person + ", numarul " + number
                corrections.append({"message": message, "paragraph_index": paragraph_index, "title": "Acord subiect - predicat", "word_index": word_index})
                last = word_index + 1
                return last
    
    return last

def check_copulative_verb(S, P, spn, corrections, last, text, paragraph_index):
    #check copulativ verb with subject
    has_cop = False
    for child in P.children:
        if child.dep_ == "cop":
            if child.tag_[:3] == "Vmi" or child.tag_[:3] == "Vai":
                spv = get_verb(child.tag_)

                if is_collective_subject(S, spv, exceptions):
                    return last
                
                if spn[:2] == spv:
                    return last
                if spn[0] == spv[0] and spv[1] == "":
                    return last
                if spn[1] == spv[1] and spv[0] == "-":
                    return last

                #it is not correct
                word_index = get_index(text, last, child.text)
                person, number, gender = get_person_number_gender(spn)
                message = "Predicatul trebuie sa fie la persoana " + person + ", numarul " + number
                corrections.append({"message": message, "paragraph_index": paragraph_index, "title": "Acord subiect - predicat", "word_index": word_index})
                last = word_index
                return last
            else:
                has_cop = True
    if has_cop == True:
        for child1 in P.children:
            if child1.dep_ == "aux":
                if child1.tag_ == "Vanp":
                    return last
                aux = get_verb(child1.tag_)
                if is_collective_subject(S, aux, exceptions):
                    return last

                if aux == spn[:2]:
                    return last
                if aux[0] == spn[0] and aux[1] == "":
                    return last

                #it is not correct
                word_index = get_index(text, last, child1.text)
                person, number, gender = get_person_number_gender(spn)
                message = "Verbul copulativ trebuie sa fie la persoana " + person + ", numarul " + number
                corrections.append({"message": message, "paragraph_index": paragraph_index, "title": "Acord subiect - predicat", "word_index": word_index})
                last = word_index
                return last
    return last

def check_predicative_name(NP, spn, corrections, last, text, paragraph_index):
    if is_job_noun(spn, NP, exceptions):
        return last

    if NP.tag_[:3] == "Rgp":
        return last

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
       return last
    #Subject is a personal pronoun without gender
    if np[1] == spn[1] and spn[2] == "-":
       return last

    #it is not correct
    person, number, gender = get_person_number_gender(spn)
    word_index = get_index(text, last, NP.text)
    message = "Numele predicativ trebuie sa fie la numarul " + number + ", genul " + gender
    corrections.append({"message": message, "paragraph_index": paragraph_index, "title": "Acord subiect - predicat", "word_index": word_index})
    last = word_index
    return last


def check_subject_and_predicate_relation(text):
    last = 0
    corrections = []
    predicates = []
    for paragraph_index, sentence in enumerate(text.split("\n")):
        doc = rom_spacy(sentence)
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
                        last = check_verbal_predicate(S, single_predicate, spn, corrections, last, text, paragraph_index)
                    else:
                        last = check_copulative_verb(S, single_predicate, spn, corrections, last, text, paragraph_index)
                        last = check_predicative_name(single_predicate, spn, corrections, last, text, paragraph_index)
    return corrections

def check_noun_and_adjective_relation(text):
    last = 0
    corrections = []
    for paragraph_index, sentence in enumerate(text.split("\n")):
        doc = rom_spacy(sentence)
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
                        #it is not correct
                        person, number, gender = get_person_number_gender(spn)
                        word_index = get_index(text, last, A.text)
                        message = "Adjectivul trebuie sa fie la numarul " + number + ", genul " + gender
                        corrections.append({"message": message, "paragraph_index": paragraph_index, "title": "Acord substantiv - adjectiv", "word_index": word_index})
                        last = word_index
    return corrections

def check_noun_and_numeral_relation(text):
    last = 0
    corrections = []
    for paragraph_index, sentence in enumerate(text.split("\n")):
        doc = rom_spacy(sentence)
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
                        person, number, gender = get_person_number_gender(spn)
                        word_index = get_index(text, last, Nr.text)
                        message = "Numeralul trebuie sa fie la numarul " + number + ", genul " + gender
                        corrections.append({"message": message, "paragraph_index": paragraph_index, "title": "Acord substantiv - numeral", "word_index": word_index})
                        last = word_index
    return corrections

def check_noun_and_unstated_article_relation(text):
    last = 0
    corrections = []
    for paragraph_index, sentence in enumerate(text.split("\n")):
        doc = rom_spacy(sentence)
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
                        person, number, gender = get_person_number_gender(spn)
                        word_index = get_index(text, last, token.text)
                        message = "Articolul nehotarat trebuie sa fie la numarul " + number + ", genul " + gender
                        corrections.append({"message": message, "paragraph_index": paragraph_index, "title": "Acord substantiv - acord nehotarat", "word_index": word_index})
                        last = word_index
                elif token.tag_[:3] == "Di3":
                    UA = token.head
                    N = token.head
                    spn = get_noun(N.tag_)
                    if spn[2] == "p":
                        continue
                    else:
                        #it is not correct
                        person, number, gender = get_person_number_gender(spn)
                        word_index = get_index(text, last, token.text)
                        message = "Articolul nehotarat trebuie sa fie la numarul " + number + ", genul " + gender
                        corrections.append({"message": message, "paragraph_index": paragraph_index, "title": "Acord substantiv - acord nehotarat", "word_index": word_index})
                        last = word_index

    return corrections


def get_index(text, last, token):
    import string

    strip = lambda s: "".join([c for c in s if c not in string.punctuation or c == '-'])
    text = [strip(word.text) for word in rom_spacy(text)]
    # text = [strip(word) for word in split_text(text)]
    return last + text[last:].index(token)

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
            message = "Reformulare. Expresia (" + item["mistake"] + ") este o cacofonie."
            paragraph_index = item["paragraph"]
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
        for item in result:
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
            output_list.append(create_output(case, message, paragraph_index, word_index))
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
    for error in output['correction']:
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
    for error in output['correction']:
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


def change_format(output):
    for error in output["correction"]:
        if isinstance(error["word_index"], int):
            error["word_index"] = [error["word_index"]]
    
    fix_dashes(output)
    fix_punctuation(output)
    mistakes = []
    for error in output["correction"]:
        par_offset = get_paragraph_offset(error["paragraph_index"], output["split_text"])
        word_offset_start = get_word_offset(error["word_index"][0], output["split_text"][error["paragraph_index"]])
        word_offset_end = get_word_offset(error["word_index"][-1], output["split_text"][error["paragraph_index"]], True)
        error["correction_index"] = [par_offset + word_offset_start, par_offset + word_offset_end]
    return output
        

def identify_mistake(sentence):

    doc = rom_spacy(sentence)
    
    # for token in doc:
    #     print(token, token.tag_, token.head, token.dep_)
    
    output = {}
    output["split_text"] = split_text(sentence)
    output_list = []
    correct = True
    #Step1
    result = check_cacophony(sentence.lower())
    if (result != []):
        correct = False
        output_list = create_output_list(result, "Cacofonie")
    
    #Step2 - ca si
    result = check_si_parazitar(sentence)
    if (result != []):
        correct = False
        output_list += create_output_list(result, "şi parazitar")
        
    #Step3 - Repetitions
    result = check_repetitions(sentence.lower())
    if (result != {}):
        correct = False
        output_list += create_output_list(result, "Repetiţie")
    
    #Step4 - Conjunctii coordonatoare
    result = check_coordinative_conjunctions(sentence)
    if result != []:
        correct = False
        output_list += create_output_list(result, "Conjuncţie coordonatoare", corrects_coordonative_error(result, sentence))

    #Step5 - Conjunctii adversative
    result = check_adversative_conjunctions(sentence)
    if result != []:
        correct = False
        output_list += create_output_list(result, "Conjuncţie adversativă", corrects_adversative_error(result, sentence))

    #Step6 - Repetitie sinonime
    result = check_synonyms_repetitions(sentence.lower())
    if result != {}:
        correct = False
        output_list += create_output_list(result, "Repetiţie sinonime")
    
    #Step7 - Voi
    result = check_voi_verb(sentence)
    if result != []:
        correct = False
        output_list += create_output_list(result, "Verbul 'a voi'", corrects_voi_verb(result, sentence))
    
    #Step8 - Comparativ de superioritate
    result = check_comparative(sentence)
    if result != []:
        correct = False
        output_list += create_output_list(result, "Comparativ de superioritate", corrects_comparative(result, sentence))

    #Step9 - Totusi
    result = check_totusi(sentence)
    if result != []:
        correct = False
        output_list += create_output_list(result, "Totuşi", corrects_coordonative_error(result, sentence))
    
    #Step10 - Adverbe la inceput
    result = check_adverbs_at_the_beginning(sentence)
    if result != []:
        correct = False
        output_list += create_output_list(result, "Adverbe la începutul propoziţiei", corrects_adverbs_at_the_beginning(result, sentence))

    #Step11 - Adverbe intercalate
    result = check_adverbs_in_middle(sentence)
    if result != []:
        correct = False
        output_list += create_output_list(result, "Adverbe intercalate", corrects_adverbs_in_middle(result, sentence))

    #Step12 - Subject and predicate
    output_list.extend(check_subject_and_predicate_relation(sentence))

    #Step13 - Noun and adjective relation
    output_list.extend(check_noun_and_adjective_relation(sentence))

    #Step14 - Noun and numeral relation
    output_list.extend(check_noun_and_numeral_relation(sentence))

    #Step15 - Noun and unstated article relation
    output_list.extend(check_noun_and_unstated_article_relation(sentence))

    output["correction"] = output_list
    output = change_format(output)

    return output


def correct_text_ro(text):
    p = re.compile('\s\s+')
    text = p.sub(' ', text)
    paragraphs = text.split('\n')
    output = {'split_text': [], 'correction': []}
    for i, paragraph in enumerate(paragraphs):
        mistakes = identify_mistake(paragraph)
        output['split_text'].append(mistakes['split_text'][0])
        for error in mistakes['correction']:
            error['paragraph_index'] = i
            output['correction'].append(error)
    
    return output
