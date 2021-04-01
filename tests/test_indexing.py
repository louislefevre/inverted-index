import unittest

from inverted_index.indexing import InvertedIndex, Posting


class TestInvertedIndex(unittest.TestCase):
    def setUp(self):
        self.documents = {1: "First document with text.",
                          2: "Second document also with text.",
                          3: "Third doc that might have text...",
                          4: "Fourth one that definitely has some words.",
                          5: "Last one that, just like the first doc, has word words."}
        self.collection = {1: ['first', 'document', 'text'],
                           2: ['second', 'document', 'also', 'text'],
                           3: ['third', 'doc', 'might', 'text'],
                           4: ['fourth', 'one', 'definit', 'word'],
                           5: ['last', 'one', 'like', 'first', 'doc', 'word', 'word']}
        self.postings = {'first': {1: Posting(freq=1, tfidf=0.0, positions=[0]),
                                   5: Posting(freq=1, tfidf=0.0, positions=[3])},
                         'document': {1: Posting(freq=1, tfidf=0.0, positions=[1]),
                                      2: Posting(freq=1, tfidf=0.0, positions=[1])},
                         'text': {1: Posting(freq=1, tfidf=0.0, positions=[2]),
                                  2: Posting(freq=1, tfidf=0.0, positions=[3]),
                                  3: Posting(freq=1, tfidf=0.0, positions=[3])},
                         'second': {2: Posting(freq=1, tfidf=0.0, positions=[0])},
                         'also': {2: Posting(freq=1, tfidf=0.0, positions=[2])},
                         'third': {3: Posting(freq=1, tfidf=0.0, positions=[0])},
                         'doc': {3: Posting(freq=1, tfidf=0.0, positions=[1]),
                                 5: Posting(freq=1, tfidf=0.0, positions=[4])},
                         'might': {3: Posting(freq=1, tfidf=0.0, positions=[2])},
                         'fourth': {4: Posting(freq=1, tfidf=0.0, positions=[0])},
                         'one': {4: Posting(freq=1, tfidf=0.0, positions=[1]),
                                 5: Posting(freq=1, tfidf=0.0, positions=[1])},
                         'definit': {4: Posting(freq=1, tfidf=0.0, positions=[2])},
                         'word': {4: Posting(freq=1, tfidf=0.0, positions=[3]),
                                  5: Posting(freq=2, tfidf=0.0, positions=[5, 6])},
                         'last': {5: Posting(freq=1, tfidf=0.0, positions=[0])},
                         'like': {5: Posting(freq=1, tfidf=0.0, positions=[2])}
                         }
        self.vocab = self.postings.keys()

        self.inv_index = InvertedIndex()
        self.inv_index.parse(self.documents)

    def test_len(self):
        self.assertEqual(len(self.inv_index), len(self.vocab))

    def test_getitem(self):
        for term, postings in self.postings.items():
            self.assertDictEqual(postings, self.inv_index[term].postings)

    def test_setitem(self):
        for term in self.vocab:
            self.assertIsNotNone(self.inv_index[term])
            self.inv_index[term] = None
            self.assertIsNone(self.inv_index[term])

    def test_delitem(self):
        for term in self.vocab:
            self.assertIn(term, self.inv_index)
            del self.inv_index[term]
            self.assertNotIn(term, self.inv_index)

    def test_contains(self):
        for term in self.vocab:
            self.assertTrue(term in self.inv_index)
            self.assertFalse(term not in self.inv_index)

    def test_clear(self):
        self.assertTrue(self.inv_index)
        self.inv_index.clear()
        self.assertFalse(self.inv_index)

    def test_index(self):
        for term in self.vocab:
            self.assertIn(term, self.inv_index.index)

    def test_collection(self):
        for doc_id, tokens in self.collection.items():
            self.assertListEqual(tokens, self.inv_index.collection[doc_id])

    def test_words(self):
        pass

    def test_vocab(self):
        self.assertListEqual(sorted(self.vocab), self.inv_index.vocab)

    def test_collection_length(self):
        self.assertEqual(len(self.collection), self.inv_index.collection_length)

    def test_word_count(self):
        pass

    def test_avg_length(self):
        pass

    def test_word_counter(self):
        pass
