from typing import Callable, List, Dict
from overrides import overrides
import re, sys
from collections import Counter
from antu.io.instance import Instance
from antu.io.fields.field import Field
from antu.io.fields.text_field import TextField
from antu.io.fields.index_field import IndexField
from antu.io.token_indexers.token_indexer import TokenIndexer
from antu.io.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from antu.io.dataset_readers.dataset_reader import DatasetReader
from utils.atom_ccg_indexes import AtomCCGIndexer


class PTBReader(DatasetReader):
    def __init__(self, field_list: List[str]):
        self.field_list = field_list

    def _isfloat(self, s):
        try:
            float(s)
            return True
        except:
            return False

    def _read(self, file_path:str):
        with open(file_path, 'r') as fp:
            for line in fp:
                orign_word = []
                word_sent = []
                ccg_sent = []
                token_list = line.strip().split(' ')
                for token in token_list:
                    tok = token.split('|')
                    word = tok[0]
                    '''if '-' or '/' in tok[0]:
                        word = re.sub('\d', '0', word)
                    tempstr = word.replace(',', '')
                    if self._isfloat(tempstr):
                        word = '0'''
                    word_sent.append(word)
                    ccg_sent.append(tok[2])
                    orign_word.append(tok[0])
                tokens = (orign_word, word_sent, ccg_sent)
                if len(tokens) > 1:
                    yield tokens

    @overrides
    def read(self, file_path: str) -> List[Instance]:
        # Build indexers
        indexers = dict()
        word_indexer = SingleIdTokenIndexer(
            ['word'], (lambda x: x.casefold()))
        indexers['word'] = [word_indexer, ]
        tag_indexer = SingleIdTokenIndexer(['tag'])
        atom_ccg = AtomCCGIndexer(['atom_ccg'])
        indexers['tag'] = [tag_indexer, atom_ccg]

        # Build instance list
        res = []
        for sentence in self._read(file_path):
            res.append(self.input_to_instance(sentence, indexers))
        return res

    @overrides
    def input_to_instance(
            self,
            inputs: List[List[str]],
            indexers: Dict[str, List[TokenIndexer]]) -> Instance:
        fields = []
        if 'word' in self.field_list:
            fields.append(TextField('word', inputs[0], indexers['word']))
        if 'tag' in self.field_list:
            fields.append(TextField('tag', inputs[2], indexers['tag']))
        return Instance(fields)
