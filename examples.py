from rb.parser.spacy_parser import SpacyParser
from rb.core.lang import Lang
from rb.core.document import Document

nlp_ro = SpacyParser.get_instance().get_model(Lang.EN)

test_text_ro = "I was going to the store, when an elephant appeared in 23.2 seconds. I was going to shop."

# tokenize
docs_ro = nlp_ro(test_text_ro)
# print all attributes of token objects
print(dir(docs_ro[0]))

for token in docs_ro:
    print(token.lemma_, token.is_stop, token.tag_, token.pos_)

# # eng
# docs_en = Document(Lang.EN, 'This is a sample document. It can contain multiple sentences and paragraphs')
# docs_en.indices.