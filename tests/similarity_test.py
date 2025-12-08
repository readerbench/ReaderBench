import unittest
from rb.core.document import Document
from rb.core.lang import Lang
from rb.similarity.word2vec import Word2Vec
from rb.similarity.lda import LDA
from rb.similarity.lsa import LSA
from rb.similarity.aoa import AgeOfAcquisition

class SimilarityTest(unittest.TestCase):

    def test_load_word2vec(self):
        w2v = Word2Vec('coca', Lang.EN)
        self.assertEqual(len(list(w2v.vectors.values())[0]), 300, "The dimension should have been 300")

    def test_load_lda(self):
        lda = LDA('coca', Lang.EN)
        self.assertEqual(len(list(lda.vectors.values())[0]), 300, "The dimension should have been 300")

    def test_load_lsa(self):
        lsa = LSA('coca', Lang.EN)
        self.assertEqual(len(list(lsa.vectors.values())[0]), 300, "The dimension should have been 300")

    def test_load_aoa(self):
        aoa = AgeOfAcquisition(Lang.EN)
        kuperman = aoa.get_kuperman_value("antihuman")
        self.assertEqual(kuperman, 11.08, "The kuperman value of 'antihuman' should have been 11.08")
        shock = aoa.get_shock_value("argue")
        self.assertEqual(shock, 4.24, "The shock value of 'argue' should have been 4.24")
        bird = aoa.get_bird_value("argument")
        self.assertIsNone(bird, "The bird value for 'argument' should have been None")

if __name__ == '__main__':
    unittest.main()