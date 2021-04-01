class InvertedIndex:
    def __init__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    def __missing__(self, key):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __reversed__(self):
        raise NotImplementedError

    def __contains__(self, item):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __bool__(self):
        raise NotImplementedError
