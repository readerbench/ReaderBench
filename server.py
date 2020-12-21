from __future__ import absolute_import

import json

import os
import random
import string
import sys
from pathlib import Path

import spacy

# from werkzeug import secure_filename
import uuid
from flask import (Flask, Response, abort, flash, jsonify, render_template,
                   request)
from flask_cors import CORS
from rb.core.lang import Lang
# from rb.diacritics.model_diacritice import Diacritics
from rb.parser.spacy_parser import SpacyParser
from rb.utils.downloader import download_model
from rb.utils.rblogger import Logger
from rb.utils.utils import str_to_lang
from rb.core.text_element_type import TextElementType
from rb.utils.downloader import download_scoring, download_classifier

app = Flask(__name__)
CORS(app)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27123d441f2b6176a'
app.config['UPLOAD_FOLDER'] = '.'

spacyInstance = SpacyParser()
logger = Logger.get_logger()

@app.route('/spacy', methods=['POST'])
def create_spacy_doc():
    doc = spacyInstance.process(request.json['doc'])
    response = jsonify({'doc': doc})
    return response, 200

@app.route('/isalive', methods=['GET'])
def check_connection():
    response = jsonify("alive")
    return response, 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8087, debug=False)
