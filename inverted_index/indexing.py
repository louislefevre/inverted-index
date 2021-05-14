import copy
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator, Union

from itertools import chain


class InvertedIndex:
    def __init__(self):
        self._index: dict[str, 'PostingList'] = {}
        self._documents: dict[str, list[str]] = {}

    def __iter__(self) -> Iterator[str]:
        return iter(self._index)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._index

    def __len__(self) -> int:
        return len(self._index)

    def __bool__(self) -> bool:
        return bool(self._index)

    def __repr__(self) -> str:
        return repr(self._index)

    def __str__(self) -> str:
        return ''.join([f'{key}: {str(value)}\n' for key, value in self._index.items()])

    @property
    def index(self) -> dict[str, 'PostingList']:
        return self._index

    @property
    def documents(self) -> dict[str, list[str]]:
        return self._documents

    def words(self, doc_id: str = None, sort=False) -> list[str]:
        words = self._documents[doc_id] if doc_id else list(chain.from_iterable(self._documents.values()))
        if sort:
            return sorted(words)
        return words

    def vocab(self, doc_id: str = None, sort=False) -> list[str]:
        vocab = set(self._documents[doc_id]) if doc_id else self._index
        if sort:
            return sorted(vocab)
        return list(vocab)

    def word_count(self, doc_id: str = None) -> int:
        words = self.words(doc_id=doc_id)
        return len(words)

    def vocab_count(self, doc_id: str = None) -> int:
        vocab = self.vocab(doc_id=doc_id)
        return len(vocab)

    def document_count(self) -> int:
        return len(self._documents)

    def avg_length(self) -> float:
        return len(self.words()) / len(self._documents)

    def word_counter(self, doc_id: str = None) -> Counter[str]:
        return Counter(self.words(doc_id=doc_id))

    def add(self, doc_id: str, tokens: list[str]) -> None:
        self._documents[doc_id] = tokens
        for pos, term in enumerate(tokens):
            if term not in self._index:
                self._index[term] = PostingList()
            self._index[term].add(doc_id, pos)

    def get(self, term: str) -> 'PostingList':
        return self._index[term]

    def remove(self, term: str) -> None:
        if term not in self._index:
            raise KeyError(f"Missing key '{term}' - term is not present")
        del self._index[term]

    def pop(self, term: str) -> Union['PostingList', None]:
        return self._index.pop(term, None)

    def purge(self, doc_id: str) -> None:
        for word in self._documents[doc_id]:
            inv_list = self._index[word]
            inv_list.remove(doc_id)
            if not inv_list:
                del self._index[word]
        del self._documents[doc_id]

    def clear(self) -> None:
        self._index.clear()

    def clone(self) -> 'InvertedIndex':
        return copy.deepcopy(self)

    def update(self, index: 'InvertedIndex') -> None:
        self._index.update(index.index)

    def merge(self, index: 'InvertedIndex') -> None:
        return NotImplementedError


class PostingList:
    def __init__(self):
        self._postings: dict[str, 'Posting'] = dict()

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._postings

    def __len__(self) -> int:
        return len(self._postings)

    def __bool__(self) -> bool:
        return bool(self._postings)

    def __repr__(self) -> str:
        return repr(self._postings)

    def __str__(self) -> str:
        return ''.join([f'{key}: ({value.freq, value.tfidf, value.positions})\n'
                        for key, value in self._postings.items()])

    @property
    def postings(self) -> dict[str, 'Posting']:
        return self._postings

    def doc_freq(self) -> int:
        return len(self)

    def add(self, doc_id: str, pos: int) -> None:
        if doc_id not in self._postings:
            self._postings[doc_id] = Posting()
        posting = self._postings[doc_id]
        posting.freq += 1
        posting.positions.append(pos)

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._postings:
            raise KeyError(f"Missing key '{doc_id}' - document is not present")
        del self._postings[doc_id]

    def get(self, doc_id: str) -> 'Posting':
        return self._postings[doc_id]

    def clear(self) -> None:
        self._postings.clear()

    def clone(self) -> 'PostingList':
        return copy.deepcopy(self)


@dataclass
class Posting:
    freq: int = 0
    tfidf: float = 0.0
    positions: list[int] = field(default_factory=list)
