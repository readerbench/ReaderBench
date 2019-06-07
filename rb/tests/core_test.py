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


if __name__ == '__main__':
    unittest.main()