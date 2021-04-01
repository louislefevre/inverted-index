import itertools
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from inverted_index.textparser import clean_text


class InvertedIndex:
    def __init__(self):
        self._index: dict[str, 'InvertedList'] = {}
        self._collection: dict[int, list[str]] = {}

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

    def parse(self, documents: dict[int, str]):
        for doc_id, text in documents.items():
            tokens = clean_text(text)
            self._collection[doc_id] = tokens
            for pos, term in enumerate(tokens):
                if term not in self._index:
                    self._index[term] = InvertedList()
                self._index[term].add_posting(doc_id, pos)

    def clear(self):
        self._index.clear()

    @property
    def index(self) -> dict[str, 'InvertedList']:
        return {key: value for key, value in sorted(self._index.items())}

    @property
    def collection(self) -> dict[int, list[str]]:
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
        self._postings: dict[int, 'Posting'] = dict()

    def add_posting(self, doc_id: int, pos: int):
        if not self.contains_posting(doc_id):
            self._postings[doc_id] = Posting()
        posting = self._postings[doc_id]
        posting.freq += 1
        posting.positions.append(pos)

    def get_posting(self, doc_id: int) -> 'Posting':
        return self._postings[doc_id]

    def contains_posting(self, doc_id: int) -> bool:
        return doc_id in self._postings

    @property
    def postings(self) -> dict[int, 'Posting']:
        return self._postings

    @property
    def doc_freq(self) -> int:
        return len(self._postings)


@dataclass
class Posting:
    freq: int = 0
    tfidf: float = 0.0
    positions: list = field(default_factory=list)
