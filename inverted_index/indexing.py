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
    def vocab(self) -> list[str]:
        return sorted(self._index)

    @property
    def avg_length(self) -> float:
        return len(self.words) / len(self._documents)

    @property
    def word_counter(self) -> Counter[str]:
        return Counter(self.words)

    def add_document(self, doc_name: str, tokens: list[str]):
        self._documents[doc_name] = tokens
        for pos, term in enumerate(tokens):
            if term not in self._index:
                self._index[term] = InvertedList()
            self._index[term].add_posting(doc_name, pos)

    def remove_document(self, doc_name: str):
        for term in self._documents[doc_name]:
            self._index[term].remove_posting(doc_name)
            if not self._index[term]:
                del self._index[term]
        del self._documents[doc_name]

    def get_doc_postings(self, doc_name: str) -> list['Posting']:
        return [self._index[term].get_posting(doc_name) for term in self._documents[doc_name]]

    def get_term_postings(self, term: str) -> dict[str, 'Posting']:
        return self._index[term].postings

    def clear(self):
        self._index.clear()


class InvertedList:
    def __init__(self):
        self._postings: dict[str, 'Posting'] = dict()

    def __contains__(self, doc_name: str) -> bool:
        return doc_name in self._postings

    def __bool__(self):
        return bool(self._postings)

    @property
    def postings(self) -> dict[str, 'Posting']:
        return self._postings

    @property
    def doc_freq(self) -> int:
        return len(self._postings)

    def add_posting(self, doc_name: str, pos: int):
        if doc_name not in self._postings:
            self._postings[doc_name] = Posting()
        posting = self._postings[doc_name]
        posting.freq += 1
        posting.positions.append(pos)

    def remove_posting(self, doc_name: str):
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
