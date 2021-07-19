from collections import OrderedDict



NULL = 'Null'


class VocabDict:
    def __init__(self, tokens, max_token_length=None, manual_tokens=None, add_null=True):
        self.__max_token_length = max_token_length
        if self.__max_token_length is None:
            self.__max_token_length = max([len(t) for t in tokens])

        if manual_tokens:
            for t in manual_tokens:
                tokens.append(t)

        self.__add_null = add_null
        self.__tok2id = OrderedDict()
        if add_null:
            self.__tok2id[NULL] = 0

        for t in tokens:
            t = self.__truncate(t)
            if t not in self.__tok2id:

                self.__tok2id[t] = len(self.__tok2id)

        self.__id2tok = OrderedDict({i: t for (t, i) in self.__tok2id.items()})

    def has(self, token):
        token = self.__truncate(token)
        if isinstance(token, int):
            return token in self.__id2tok
        elif isinstance(token, str):
            return token in self.__tok2id
        else:
            raise TypeError(f"VocabDict only accepts str or int. "
                            f"{type(token)} was given.")

    def to_id(self, token, str_cast=False):
        if str_cast:
            token = str(token)

        token = self.__truncate(token)

        return self.__tok2id[token]

    def to_id_list(self, tokens, str_cast=False):
        return [self.to_id(t, str_cast) for t in tokens]

    def to_id_batch(self, list_of_tokens, str_cast=False):
        return [self.to_id_list(lst, str_cast) for lst in list_of_tokens]

    def to_str(self, id, int_cast=False):
        if int_cast:
            id = int(id)
        return self.__id2tok[id]

    def to_str_list(self, ids, int_cast=False):
        return [self.to_str(i, int_cast=int_cast) for i in ids]

    def to_str_batch(self, list_of_ids, int_cast=False):
        return [self.to_str_list(lst, int_cast) for lst in list_of_ids]

    def __truncate(self, t):
        return t[:self.max_token_length]

    @property
    def tok2id(self):
        return self.__tok2id

    @property
    def id2tok(self):
        return self.__id2tok

    @property
    def size(self):
        return len(self.__tok2id)

    @property
    def is_empty(self):
        return self.size == 0

    @property
    def max_token_length(self):
        return self.__max_token_length

    @property
    def null_id(self):
        return self.to_id(NULL)
