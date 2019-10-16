from __future__ import absolute_import

import json

import os
import random
import string
import sys
from pathlib import Path

import spacy

from werkzeug import secure_filename
import uuid
from flask import (Flask, Response, abort, flash, jsonify, render_template,
                   request)
from rb.ro_corrections.ro_correct import correct_text_ro
from rb.core.lang import Lang
from rb.diacritics.model_diacritice import Diacritics
from rb.parser.spacy_parser import SpacyParser
from rb.utils.downloader import download_model
from rb.processings.scoring.essay_scoring import EssayScoring
from rb.processings.fluctuations.fluctuations import Fluctuations
from rb.utils.rblogger import Logger

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27123d441f2b6176a'
app.config['UPLOAD_FOLDER'] = '.'

spacyInstance = SpacyParser()
logger = Logger.get_logger()

@app.route('/test', methods=['POST'])
def handle_get():
    print(request.json)
    result = {"doc": {"blocks": [{"block": "Ana are mere", "sentences": [{}]}]}}
    response = jsonify(result)
    return response, 200

@app.route('/spacy', methods=['POST'])
def create_spacy_doc():
    doc = spacyInstance.process(request.json['doc'])
    response = jsonify({'doc': doc})
    return response, 200

@app.route('/isalive', methods=['GET'])
def check_connection():
    response = jsonify("alive")
    return response, 200

@app.route('/ro_correct', methods=['POST'])
def ro_correct():
    data = request.get_json()
    text = data['text']

    #text = text.replace('    ', '\t').replace('\t', '\n')
    res = correct_text_ro(text)
    # for error in res:
    #     print(f'error {error["title"]}: ')
    #     for indices in error['correction_index']:
    #         print(text[indices[0]: indices[1] + 1])
    for mistake in res['corrections']:
        print(f'mistake: {mistake["mistake"]}')
        for m in mistake['index']:
            print(res['split_text'][m[0]][m[1]])

    return jsonify(res)

@app.route('/scoring', methods=['POST'])
def scoring():
    data = request.get_json()
    text = data['text']
    essay_scoring = EssayScoring()
    score = essay_scoring.predict(text, file_to_svr_model='svr_gamma.p')
    return jsonify(str(score))

@app.route('/fluctuations', methods=['POST'])
def fluctuations():
    data = request.get_json()
    text = data['text']
    if 'lang' in data:
        lang = data['lang']
    else:
        lang = 'ro'
    fl = Fluctuations()

    if lang == 'ro':
        res = fl.compute_indices(text, lang=Lang.RO)
    else:
        res = fl.compute_indices(text, lang=Lang.EN)

    return jsonify(res)

@app.route('/diacritics', methods=['POST'])
def restore_diacritics():
    logger.info('Received {} request for {}'.format(request.method, request.path))

    text = request.json['doc']
    tmp_in_file = "".join([random.choice(string.ascii_uppercase + string.digits) for _ in range(10)])
    tmp_out_file = "".join([random.choice(string.ascii_uppercase + string.digits) for _ in range(10)])
    with open(tmp_in_file, 'w', encoding='utf-8') as f:
        f.write(text)

    diacritics = Diacritics()
    diacritics.args.use_window_characters = True
    diacritics.args.use_dummy_word_embeddings = True
    diacritics.args.use_sentence_embedding = False
    diacritics.args.use_word_embedding = False
    diacritics.args.buffer_size_shuffle = 1000 
    diacritics.args.load_model_name = 'resources/ro/models/chars.h5'
    model_file = Path(diacritics.args.load_model_name)
    if not (model_file.exists() and model_file.is_file()):
        download_model(Lang.RO, 'diacritics')

    diacritics.args.nr_classes = 4
    diacritics.args.use_tags = False
    diacritics.args.use_deps = False
    diacritics.args.restore_diacritics = tmp_in_file
    diacritics.args.output_file_restore = tmp_out_file
    diacritics.args.batch_size_restore = 256

    diacritics.args.save_model = False
    diacritics.args.do_test = False
    diacritics.args.run_train_validation = False

    diacritics.run_task()
    with open(tmp_out_file, 'r', encoding='utf-8') as content_out:
        content = content_out.read()
    os.remove(tmp_in_file)
    os.remove(tmp_out_file)
    mistake_intervals = []
    for i, c in enumerate(content):
        if c != text[i]:
            st, dr = diacritics.get_word_interval_mistake(content, i) # [st, dr]
            info_wrong_word = (st, dr, text[st:dr + 1], content[st: dr + 1]) 
            if info_wrong_word not in mistake_intervals:
                mistake_intervals.append(info_wrong_word)
    response = jsonify({'restored': content, 'mistakes': mistake_intervals})
    return response, 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8082, debug=False)
