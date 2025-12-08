import unittest
from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.pos import POS
from rb.cna.cna_graph import CnaGraph

class CnaTest(unittest.TestCase):

    def test_create_graph_en(self):
        text_string = "This is a text string. Does the parsing work?"
        doc = Document(Lang.EN, text_string)
        cna = CnaGraph(doc)
        self.assertEqual(len(cna.graph.edges), 4, "Should be 4")

    

if __name__ == '__main__':
    unittest.main()