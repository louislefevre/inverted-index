import copy
from collections import Counter
from typing import Iterator, Union, Hashable

from itertools import chain


class InvertedIndex:
    def __init__(self):
        self._index: dict[Hashable, 'PostingList'] = {}
        self._documents: dict[Hashable, list[str]] = {}

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._index)

    def __contains__(self, doc_id: Hashable) -> bool:
        return doc_id in self._index

    def __len__(self) -> int:
        return len(self._index)

    def __bool__(self) -> bool:
        return bool(self._index)

    def __repr__(self) -> str:
        return repr(self._index)

    def __str__(self) -> str:
        return ''.join([f'{key}: {str(value)}' for key, value in self._index.items()])

    @property
    def index(self) -> dict[Hashable, 'PostingList']:
        return self._index

    @property
    def documents(self) -> dict[Hashable, list[str]]:
        return self._documents

    def words(self, doc_id: Hashable = None, sort=False) -> list[str]:
        words = self._documents[doc_id] if doc_id else list(chain.from_iterable(self._documents.values()))
        if sort:
            return sorted(words)
        return words

    def vocab(self, doc_id: Hashable = None, sort=False) -> list[str]:
        vocab = set(self._documents[doc_id]) if doc_id else self._index
        if sort:
            return sorted(vocab)
        return list(vocab)

    def word_count(self, doc_id: Hashable = None) -> int:
        words = self.words(doc_id=doc_id)
        return len(words)

    def vocab_count(self, doc_id: Hashable = None) -> int:
        vocab = self.vocab(doc_id=doc_id)
        return len(vocab)

    def document_count(self) -> int:
        return len(self._documents)

    def average_length(self) -> float:
        return len(self.words()) / len(self._documents)

    def word_counter(self, doc_id: Hashable = None) -> Counter[str]:
        return Counter(self.words(doc_id=doc_id))

    def add(self, doc_id: Hashable, tokens: list[str], track_positions=False) -> None:
        self._documents[doc_id] = tokens
        for pos, term in enumerate(tokens):
            if term not in self._index:
                self._index[term] = PostingList()
            self._index[term]._add(doc_id, pos, track_positions=track_positions)

    def get(self, term: str) -> 'PostingList':
        return self._index[term]

    def remove(self, term: str) -> None:
        if term not in self._index:
            raise KeyError(f"Missing key '{term}' - term is not present")
        del self._index[term]

    def pop(self, term: str) -> Union['PostingList', None]:
        return self._index.pop(term, None)

    def purge(self, doc_id: Hashable) -> None:
        for word in self._documents[doc_id]:
            inv_list = self._index[word]
            inv_list._remove(doc_id)
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
        for term, posting_list in index.index.items():
            if term in self._index:
                self._index[term]._merge(posting_list)
            else:
                self._index[term] = posting_list


class PostingList:
    def __init__(self):
        self._postings: dict[Hashable, 'Posting'] = dict()

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._postings)

    def __contains__(self, doc_id: Hashable) -> bool:
        return doc_id in self._postings

    def __len__(self) -> int:
        return len(self._postings)

    def __bool__(self) -> bool:
        return bool(self._postings)

    def __repr__(self) -> str:
        return repr(self._postings)

    def __str__(self) -> str:
        return ''.join([f'{key}: ({value.frequency, value.positions if not None else "[]"})\n'
                        for key, value in self._postings.items()])

    def _add(self, doc_id: Hashable, pos: int, track_positions=False) -> None:
        if doc_id not in self._postings:
            self._postings[doc_id] = Posting()
        posting = self._postings[doc_id]
        posting._increment()
        if track_positions:
            posting._add_pos(pos)

    def _remove(self, doc_id: Hashable) -> None:
        if doc_id not in self._postings:
            raise KeyError(f"Missing key '{doc_id}' - document is not present")
        del self._postings[doc_id]

    def _update(self, posting_list: 'PostingList') -> None:
        self._postings.update(posting_list.postings)

    def _merge(self, posting_list: 'PostingList') -> None:
        for doc_id, posting in posting_list.postings.items():
            if doc_id not in self._postings:
                self._postings[doc_id] = posting

    @property
    def postings(self) -> dict[Hashable, 'Posting']:
        return self._postings

    def document_frequency(self) -> int:
        return len(self)

    def get(self, doc_id: Hashable) -> 'Posting':
        return self._postings[doc_id]

    def clone(self) -> 'PostingList':
        return copy.deepcopy(self)


class Posting:
    def __init__(self):
        self._frequency: int = 0
        self._positions: list[int] = []

    def _increment(self) -> None:
        self._frequency += 1

    def _add_pos(self, pos: int) -> None:
        self._positions.append(pos)

    @property
    def frequency(self):
        return self._frequency

    @property
    def positions(self) -> list[int]:
        return self._positions
