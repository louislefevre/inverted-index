import itertools
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Callable


class InvertedIndex:
    def __init__(self):
        self._index: dict[str, 'InvertedList'] = {}
        self._collection: dict[str, list[str]] = {}

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, item: str) -> 'InvertedList':
        return self._index[item]

    def __setitem__(self, key, value):
        self._index[key] = value

    def __delitem__(self, key):
        del self._index[key]

    def __iter__(self) -> Iterable[str]:
        return iter(self._index)

    def __contains__(self, item) -> bool:
        return item in self._index

    def __repr__(self) -> str:
        return repr(self._index)

    def __str__(self) -> str:
        return ''.join([f'{key}: {value.postings}\n' for key, value in self._index.items()])

    def __bool__(self) -> bool:
        return bool(self._index)

    def add_document(self, doc_name: str, tokens: list[str]):
        self._collection[doc_name] = tokens
        for pos, term in enumerate(tokens):
            if term not in self._index:
                self._index[term] = InvertedList()
            self._index[term].add_posting(doc_name, pos)

    def add_collection(self, documents: dict[str, list[str]]):
        for doc_name, tokens in documents.items():
            self.add_document(doc_name, tokens)

    def parse_document(self, doc_name: str, content: str, preprocessor: Callable[[str], list[str]]):
        tokens = preprocessor(content)
        self.add_document(doc_name, tokens)

    def parse_collection(self, documents: dict[str, str], preprocessor: Callable[[str], list[str]]):
        for doc_name, content in documents.items():
            self.parse_document(doc_name, content, preprocessor)

    def clear(self):
        self._index.clear()

    @property
    def index(self) -> dict[str, 'InvertedList']:
        return self._index

    @property
    def collection(self) -> dict[str, list[str]]:
        return self._collection

    @property
    def words(self) -> list[str]:
        return list(itertools.chain.from_iterable(self._collection.values()))

    @property
    def vocab(self) -> list[str]:
        return sorted(self._index)

    @property
    def collection_length(self) -> int:
        return len(self._collection)

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def vocab_count(self) -> int:
        return len(self._index)

    @property
    def avg_length(self) -> float:
        return self.word_count / self.collection_length

    @property
    def word_counter(self) -> Counter[str]:
        return Counter(self.words)


class InvertedList:
    def __init__(self):
        self._postings: dict[str, 'Posting'] = dict()

    def add_posting(self, doc_name: str, pos: int):
        if not self.contains_posting(doc_name):
            self._postings[doc_name] = Posting()
        posting = self._postings[doc_name]
        posting.freq += 1
        posting.positions.append(pos)

    def get_posting(self, doc_name: str) -> 'Posting':
        return self._postings[doc_name]

    def contains_posting(self, doc_name: str) -> bool:
        return doc_name in self._postings

    @property
    def postings(self) -> dict[str, 'Posting']:
        return self._postings

    @property
    def doc_freq(self) -> int:
        return len(self._postings)


@dataclass
class Posting:
    freq: int = 0
    tfidf: float = 0.0
    positions: list = field(default_factory=list)
