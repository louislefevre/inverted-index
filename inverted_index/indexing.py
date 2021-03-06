import copy
from collections import Counter
from typing import Iterator, Union, Hashable, Dict, List, Counter as CounterType

from itertools import chain


class InvertedIndex:
    def __init__(self):
        self._index: Dict[Hashable, 'PostingList'] = {}
        self._documents: Dict[Hashable, List[str]] = {}

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

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._index == other._index and self._documents == other._documents

    @property
    def index(self) -> Dict[Hashable, 'PostingList']:
        return self._index

    @property
    def documents(self) -> Dict[Hashable, List[str]]:
        return self._documents

    def words(self, doc_id: Hashable = None, sort=False) -> List[str]:
        words = self._documents[doc_id] if doc_id else list(chain.from_iterable(self._documents.values()))
        if sort:
            return sorted(words)
        return words

    def vocab(self, doc_id: Hashable = None, sort=False) -> List[str]:
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
        if len(self._documents) != 0:
            return len(self.words()) / len(self._documents)
        return 0.0

    def word_counter(self, doc_id: Hashable = None) -> CounterType[str]:
        return Counter(self.words(doc_id=doc_id))

    def add(self, doc_id: Hashable, tokens: List[str], track_positions=False) -> None:
        self._documents[doc_id] = tokens
        for pos, term in enumerate(tokens):
            if term not in self._index:
                self._index[term] = PostingList()
            self._index[term]._add(doc_id, pos, track_positions=track_positions)

    def get(self, term: str) -> 'PostingList':
        return self._index[term]

    def remove(self, term: str) -> None:
        del self._index[term]

    def pop(self, term: str) -> Union['PostingList', None]:
        return self._index.pop(term, None)

    def purge(self, doc_id: Hashable) -> None:
        for word in set(self._documents[doc_id]):
            posting_list = self._index[word]
            posting_list._remove(doc_id)
            if not posting_list:
                del self._index[word]
        del self._documents[doc_id]

    def clear(self) -> None:
        self._index.clear()
        self._documents.clear()

    def clone(self) -> 'InvertedIndex':
        return copy.deepcopy(self)

    def update(self, index: 'InvertedIndex') -> None:
        self._index.update(index.index)
        self._documents.update(index.documents)

    def merge(self, index: 'InvertedIndex') -> None:
        for term, posting_list in index.index.items():
            if term in self._index:
                self._index[term]._merge(posting_list)
            else:
                self._index[term] = posting_list
        self._documents.update(index.documents)


class PostingList:
    def __init__(self):
        self._postings: Dict[Hashable, 'Posting'] = dict()

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

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._postings == other._postings

    def _add(self, doc_id: Hashable, pos: int, track_positions=False) -> None:
        if doc_id not in self._postings:
            self._postings[doc_id] = Posting()
        posting = self._postings[doc_id]
        posting._increment()
        if track_positions:
            posting._add_pos(pos)

    def _remove(self, doc_id: Hashable) -> None:
        del self._postings[doc_id]

    def _update(self, posting_list: 'PostingList') -> None:
        self._postings.update(posting_list.postings)

    def _merge(self, posting_list: 'PostingList') -> None:
        for doc_id, posting in posting_list.postings.items():
            if doc_id not in self._postings:
                self._postings[doc_id] = posting

    @property
    def postings(self) -> Dict[Hashable, 'Posting']:
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
        self._positions: List[int] = []

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._frequency == other._frequency and self._positions == other._positions

    def _increment(self) -> None:
        self._frequency += 1

    def _add_pos(self, pos: int) -> None:
        self._positions.append(pos)

    @property
    def frequency(self):
        return self._frequency

    @property
    def positions(self) -> List[int]:
        return self._positions
