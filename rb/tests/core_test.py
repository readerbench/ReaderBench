import unittest
from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.pos import POS

class CoreTest(unittest.TestCase):

    def test_create_document_en(self):
        text_string = "This is a text string. Does the parsing work?"
        doc = Document(Lang.EN, text_string)
        self.assertEqual(doc.text, text_string, "Should be " + text_string)

    def test_create_document_ro(self):
        text_string = "Acesta este un text de test. Întrebarea este: merge?"
        doc = Document(Lang.RO, text_string)
        self.assertEqual(doc.text, text_string, "Should be " + text_string)

    def test_get_words(self):
        text_string = "This is a text string. Does the parsing work?\nSecond paragraph."
        doc = Document(Lang.EN, text_string)
        self.assertEqual(len(doc.get_words()), 14, "Should be 14: " + str(doc.get_words()))

    def test_coref_en(self):
        text_string = "John is a programmer. He earns big money."
        doc = Document(Lang.EN, text_string)
        for word in doc.get_words():
            if word.text != "He":
                continue
            else:
                self.assertEqual(True, word.in_coref, "Should be: True")
                self.assertEqual("John", word.coref_clusters[0].main.get_root().text, "Should be: John")
        

if __name__ == '__main__':
    unittest.main()