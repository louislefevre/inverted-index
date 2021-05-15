import unittest
from collections import Counter
from typing import Any

from inverted_index.indexing import InvertedIndex


def generate_index(documents: dict[Any, list[str]]):
    words = [word for words in documents.values() for word in words]
    vocab = list(set(words))
    index = InvertedIndex()
    for doc_id, text in documents.items():
        index.add(doc_id, text)
    return index, words, vocab


class TestInvertedIndex(unittest.TestCase):
    def setUp(self):
        self.documents = {1: ['the', 'penguin', 'waddled', 'through', 'the', 'snow'],
                          2: ['the', 'seal', 'saw', 'the', 'penguin', 'and', 'chased', 'him'],
                          3: ['the', 'penguin', 'managed', 'to', 'escape', 'the', 'seal']}
        self.index, self.words, self.vocab = generate_index(self.documents)

        self.missing_documents = {4: ['a', 'polar', 'bear', 'watched'],
                                  5: ['it', 'ate', 'bad', 'seals'],
                                  6: ['which', 'saved', 'all', 'penguins']}
        self.missing_index, self.missing_words, self.missing_vocab = generate_index(self.missing_documents)

    def test_len(self):
        self.assertEqual(len(self.vocab), len(self.index))

    def test_words(self):
        self.assertEqual(self.words, self.index.words())

    def test_words_with_sort(self):
        self.assertEqual(sorted(self.words), self.index.words(sort=True))

    def test_words_with_id(self):
        for doc_id, words in self.documents.items():
            self.assertEqual(self.documents[doc_id], self.index.words(doc_id=doc_id))

    def test_words_with_id_and_sort(self):
        for doc_id, words in self.documents.items():
            self.assertEqual(sorted(self.documents[doc_id]), self.index.words(doc_id=doc_id, sort=True))

    def test_words_missing_key(self):
        for doc_id in self.missing_documents:
            self.assertRaises(KeyError, self.index.words, doc_id=doc_id)

    def test_vocab(self):
        self.assertCountEqual(self.vocab, self.index.vocab())

    def test_vocab_with_sort(self):
        self.assertEqual(sorted(self.vocab), self.index.vocab(sort=True))

    def test_vocab_with_id(self):
        for doc_id, words in self.documents.items():
            self.assertCountEqual(list(set(self.documents[doc_id])), self.index.vocab(doc_id=doc_id))

    def test_vocab_with_id_and_sort(self):
        for doc_id, words in self.documents.items():
            self.assertEqual(sorted(set(self.documents[doc_id])), self.index.vocab(doc_id=doc_id, sort=True))

    def test_vocab_missing_key(self):
        for doc_id in self.missing_documents:
            self.assertRaises(KeyError, self.index.vocab, doc_id=doc_id)

    def test_word_count(self):
        self.assertEqual(len(self.words), self.index.word_count())

    def test_word_count_with_id(self):
        for doc_id, words in self.documents.items():
            self.assertEqual(len(self.documents[doc_id]), self.index.word_count(doc_id=doc_id))

    def test_word_count_missing_key(self):
        for doc_id in self.missing_documents:
            self.assertRaises(KeyError, self.index.word_count, doc_id=doc_id)

    def test_vocab_count(self):
        self.assertEqual(len(self.vocab), self.index.vocab_count())

    def test_vocab_count_with_id(self):
        for doc_id, words in self.documents.items():
            self.assertEqual(len(set(self.documents[doc_id])), self.index.vocab_count(doc_id=doc_id))

    def test_vocab_count_missing_key(self):
        for doc_id in self.missing_documents:
            self.assertRaises(KeyError, self.index.vocab_count, doc_id=doc_id)

    def test_document_count(self):
        self.assertEqual(len(self.documents), self.index.document_count())

    def test_average_length(self):
        self.assertEqual((len(self.words) / len(self.documents)), self.index.average_length())

    def test_average_length_empty(self):
        self.index.clear()
        self.assertEqual(0.0, self.index.average_length())

    def test_word_counter(self):
        self.assertEqual(Counter(self.words), self.index.word_counter())

    def test_word_counter_with_id(self):
        for doc_id, words in self.documents.items():
            self.assertEqual(Counter(self.documents[doc_id]), self.index.word_counter(doc_id=doc_id))

    def test_word_counter_missing_key(self):
        for doc_id in self.missing_documents:
            self.assertRaises(KeyError, self.index.word_counter, doc_id=doc_id)

    def test_add(self):
        for doc_id, words in self.missing_documents.items():
            self.index.add(doc_id, words)
            for word in words:
                self.assertIn(word, self.index)

    def test_get(self):
        for doc_id, words in self.documents.items():
            for word in words:
                posting_list = self.index.get(word)
                self.assertIn(doc_id, posting_list)

    def test_get_missing_key(self):
        for word in self.missing_words:
            self.assertRaises(KeyError, self.index.get, word)

    def test_remove(self):
        for term in self.vocab:
            self.index.remove(term)
            self.assertNotIn(term, self.index)

    def test_remove_missing_key(self):
        for word in self.missing_words:
            self.assertRaises(KeyError, self.index.remove, word)

    def test_pop(self):
        for term in self.vocab:
            self.assertIsNotNone(self.index.pop(term))
            self.assertNotIn(term, self.index)

    def test_pop_missing_key(self):
        for word in self.missing_words:
            self.assertIsNone(self.index.pop(word))

    def test_purge(self):
        for doc_id, words in self.documents.copy().items():
            self.index.purge(doc_id)
            del self.documents[doc_id]
            for word in words:
                if word in [w for ws in self.documents.values() for w in ws]:
                    # For words present in documents other than the one being purged
                    self.assertNotIn(doc_id, self.index.get(word).postings)
                else:
                    # For words which are not in any other documents other than the one being purged
                    self.assertNotIn(word, self.index)
            self.assertNotIn(doc_id, self.index.documents)

    def test_purge_missing_key(self):
        for doc_id in self.missing_documents:
            self.assertRaises(KeyError, self.index.purge, doc_id)

    def test_clear(self):
        self.index.clear()
        self.assertFalse(self.index)
        self.assertFalse(self.index.documents)

    def test_clone(self):
        self.assertEqual(self.index.clone(), self.index)

    def test_clone_is_deepcopy(self):
        clone = self.index.clone()
        self.assertEqual(clone, self.index)
        for doc_id, words in self.documents.items():
            clone.add(doc_id, words)
        self.assertNotEqual(clone, self.index)

    def test_update(self):
        new_index = InvertedIndex()
        for doc_id, words in self.documents.items():
            new_index.add(doc_id, sorted(words))
        self.index.update(new_index)
        self.assertEqual(new_index, self.index)

    def test_merge(self):
        self.index.merge(self.missing_index)
        for doc_id, words in self.missing_documents.items():
            for word in words:
                self.assertIn(word, self.index)
            self.assertIn(doc_id, self.index.documents)
