# some python file
import textract

"""for supported docs check:
    https://textract.readthedocs.io/en/stable/
"""

def extract_raw_text(path_to_file):
    raw_text = textract.process(path_to_file).decode('utf-8')
    return raw_text