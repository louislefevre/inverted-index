class InvertedIndex:
    def __init__(self):
        self._index: dict[str, any] = {}

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, item: str) -> dict[str, any]:
        return self._index[item]

    def __setitem__(self, key, value):
        self._index[key] = value

    def __delitem__(self, key):
        del self._index[key]

    def __iter__(self) -> iter(str):
        return iter(self._index)

    def __contains__(self, item) -> bool:
        return item in self._index

    def __repr__(self) -> str:
        return repr(self._index)

    def __str__(self) -> str:
        return ''.join([f'{key}: {value}\n' for key, value in self._index.items()])

    def __bool__(self) -> bool:
        return bool(self._index)

    def clear(self):
        self._index.clear()

    @property
    def index(self):
        return self._index
