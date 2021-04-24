import itertools
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator, Union


class InvertedIndex:
    def __init__(self):
        self._index: dict[str, 'InvertedList'] = {}
        self._documents: dict[str, list[str]] = {}

    def __iter__(self) -> Iterator[str]:
        return iter(self._index)

    def __contains__(self, doc_name: str) -> bool:
        return doc_name in self._index

    def __len__(self) -> int:
        return len(self._index)

    def __bool__(self) -> bool:
        return bool(self._index)

    def __repr__(self) -> str:
        return repr(self._index)

    def __str__(self) -> str:
        return ''.join([f'{key}: {value.postings}\n' for key, value in self._index.items()])

    @property
    def index(self) -> dict[str, 'InvertedList']:
        return self._index

    @property
    def documents(self) -> dict[str, list[str]]:
        return self._documents

    @property
    def words(self) -> list[str]:
        return list(itertools.chain.from_iterable(self._documents.values()))

    @property
    def vocab(self, sort=True) -> list[str]:
        if sort:
            return sorted(self._index)
        return list(self._index)

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def vocab_count(self) -> int:
        return len(self._index)

    @property
    def avg_length(self) -> float:
        return len(self.words) / len(self._documents)

    @property
    def word_counter(self) -> Counter[str]:
        return Counter(self.words)

    def add_document(self, doc_name: str, tokens: list[str]) -> None:
        self._documents[doc_name] = tokens
        for pos, term in enumerate(tokens):
            if term not in self._index:
                self._index[term] = InvertedList()
            self._index[term].add_posting(doc_name, pos)

    def remove_document(self, doc_name: str) -> None:
        for term in self._documents[doc_name]:
            self.remove_posting(term, doc_name)
        del self._documents[doc_name]

    def remove_posting(self, term: str, doc_name: str) -> None:
        self._index[term].remove_posting(doc_name)
        if not self._index[term]:
            del self._index[term]

    def remove_term(self, term: str) -> None:
        if term in self._index:
            del self._index[term]

    def contains_posting(self, term: str, doc_name: str) -> bool:
        return doc_name in self._index[term]

    def contains_word(self, doc_name: str, word: str) -> bool:
        return word in self._documents[doc_name]

    def postings(self, term: str) -> dict[str, 'Posting']:
        return self._index[term].postings

    def document_postings(self, doc_name: str) -> list['Posting']:
        return [self._index[term].get_posting(doc_name) for term in self._documents[doc_name]]

    def document_frequency(self, term: str) -> int:
        return self._index[term].doc_freq

    def frequency(self, term: str, doc_name: str) -> int:
        return self._index[term].get_posting(doc_name).freq

    def tfidf(self, term: str, doc_name: str) -> float:
        return self._index[term].get_posting(doc_name).tfidf

    def document_length(self, doc_name: str) -> int:
        return len(self._documents[doc_name])

    def document_vocab(self, doc_name: str, sort=True) -> list[str]:
        if sort:
            return sorted(set(self._documents[doc_name]))
        return list(set(self._documents[doc_name]))

    def clear(self) -> None:
        self._index.clear()


class InvertedList:
    def __init__(self):
        self._postings: dict[str, 'Posting'] = dict()

    def __contains__(self, doc_name: str) -> bool:
        return doc_name in self._postings

    def __bool__(self) -> bool:
        return bool(self._postings)

    @property
    def postings(self) -> dict[str, 'Posting']:
        return self._postings

    @property
    def doc_freq(self) -> int:
        return len(self._postings)

    def add_posting(self, doc_name: str, pos: int) -> None:
        if doc_name not in self._postings:
            self._postings[doc_name] = Posting()
        posting = self._postings[doc_name]
        posting.freq += 1
        posting.positions.append(pos)

    def remove_posting(self, doc_name: str) -> None:
        if doc_name in self._postings:
            del self._postings[doc_name]
        raise KeyError(f"Missing key '{doc_name}' - document does not exist")

    def pop_posting(self, doc_name) -> Union['Posting', None]:
        return self._postings.get(doc_name, None)

    def get_posting(self, doc_name: str) -> 'Posting':
        return self._postings[doc_name]


@dataclass
class Posting:
    freq: int = 0
    tfidf: float = 0.0
    positions: list = field(default_factory=list)
