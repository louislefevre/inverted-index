from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator, Union

import itertools


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

    def words(self, doc_name: str = None, sort=False) -> list[str]:
        words = self._documents[doc_name] if doc_name else list(itertools.chain.from_iterable(self._documents.values()))
        if sort:
            return sorted(words)
        return words

    def vocab(self, doc_name: str = None, sort=False) -> list[str]:
        vocab = set(self._documents[doc_name]) if doc_name else self._index
        if sort:
            return sorted(vocab)
        return list(vocab)

    def word_count(self, doc_name: str = None) -> int:
        words = self.words(doc_name=doc_name)
        return len(words)

    def vocab_count(self, doc_name: str = None) -> int:
        vocab = self.vocab(doc_name=doc_name)
        return len(vocab)

    def document_count(self) -> int:
        return len(self._documents)

    def avg_length(self) -> float:
        return len(self.words()) / len(self._documents)

    def word_counter(self, doc_name: str = None) -> Counter[str]:
        return Counter(self.words(doc_name=doc_name))

    def add(self, doc_name: str, tokens: list[str]) -> None:
        self._documents[doc_name] = tokens
        for pos, term in enumerate(tokens):
            if term not in self._index:
                self._index[term] = InvertedList()
            self._index[term].add(doc_name, pos)

    def get(self, term: str) -> 'InvertedList':
        return self._index[term]

    def remove(self, term: str) -> None:
        if term not in self._index:
            raise KeyError(f"Missing key '{term}' - term is not present")
        del self._index[term]

    def pop(self, term: str) -> Union['InvertedList', None]:
        return self._index.pop(term, None)

    def purge(self, doc_name: str) -> None:
        for word in self._documents[doc_name]:
            inv_list = self._index[word]
            inv_list.remove(doc_name)
            if not inv_list:
                del self._index[word]
        del self._documents[doc_name]

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

    def doc_freq(self) -> int:
        return len(self._postings)

    def add(self, doc_name: str, pos: int) -> None:
        if doc_name not in self._postings:
            self._postings[doc_name] = Posting()
        posting = self._postings[doc_name]
        posting.freq += 1
        posting.positions.append(pos)

    def remove(self, doc_name: str) -> None:
        if doc_name not in self._postings:
            raise KeyError(f"Missing key '{doc_name}' - document is not present")
        del self._postings[doc_name]

    def get(self, doc_name: str) -> 'Posting':
        return self._postings[doc_name]

    def clear(self) -> None:
        self._postings.clear()


@dataclass
class Posting:
    freq: int = 0
    tfidf: float = 0.0
    positions: list[int] = field(default_factory=list)
