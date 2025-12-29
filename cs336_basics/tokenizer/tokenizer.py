
from typing import Iterator, Iterable
import regex as re


class Tokenizer:
    def __init__(self, vocab:dict[int, bytes], merges:list[tuple[bytes, bytes]], speicial_tokens:list[str] | None = None) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(speicial_tokens, reverse=True) if speicial_tokens is not None else []
        self.encoding_vocab = dict((word, index) for index, word in self.vocab.items())
        vocab_size = len(self.vocab)
        for special_token in self.special_tokens:
            special_token_encoding = special_token.encode('utf-8')
            if special_token_encoding not in self.encoding_vocab:
                self.encoding_vocab[special_token_encoding] = vocab_size
                self.vocab[vocab_size] = special_token_encoding
                vocab_size += 1
        self.merges_dict = dict((merge, index) for index, merge in enumerate(self.merges))
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens:list[str] | None = None) -> "Tokenizer":
        vocab = None
        merges = None
        with open(vocab_filepath, mode='r', encoding="utf-8") as f:
            vocab_line = f.readline()
            vocab = eval(vocab_line)
        with open(merges_filepath, mode='r', encoding='utf-8') as f:
            merges_line = f.readline()
            merges = eval(merges_line)
        return cls(vocab, merges, special_tokens)
    
    # def _get_next_chunk(self, text: str) -> Iterable[bytes]:
    #     # this seems not needed. just thought we need to reduce memory usage here. 
    #     start = 0
    #     end = len(text)
    #     CHUNKSIZE = 1024*1024

    #     while start < end:
    #         if start + CHUNKSIZE +10*self.max_token_len >= end:
    #             chunk = text[start:end]
    #             start = end
    #         else:
    #             medium = text[start+CHUNKSIZE: start+CHUNKSIZE+10*self.max_token_len]
    #             new_start = start+CHUNKSIZE
    #             while new_start < start+CHUNKSIZE+2*self.max_token_len:
    #                 if len(self.special_tokens) == 0:
    #                     for pretoken_match in re.finditer(self.PAT, medium):
    #                         pretoken = pretoken_match.group()
    #                         new_start += len(pretoken)
    #                         if new_start > start+CHUNKSIZE+2*self.max_token_len:
    #                             break
    #                 else:
    #                     for split_medium in re.split('|'.join(['('+re.escape(st)+')' for st in self.special_tokens]), medium):
    #                         if split_medium is None:
    #                             continue
    #                         if split_medium in self.special_tokens:
    #                             new_start += len(split_chunk)
    #                         else:
    #                             for pretoken_match in re.finditer(self.PAT, split_medium):
    #                                 pretoken = pretoken_match.group()
    #                                 new_start += len(pretoken)
    #                                 if new_start > start+CHUNKSIZE+2*self.max_token_len:
    #                                     break
    #                         if new_start > start+CHUNKSIZE+2*self.max_token_len:
    #                                 break
    #             chunk = text[start:new_start]
    #             start = new_start

    #         # output
    #         if len(self.special_tokens) == 0:
    #             yield chunk
    #         else:
    #             for split_chunk in re.split('|'.join(['('+re.escape(st)+')' for st in self.special_tokens]), chunk):
    #                 if split_chunk is None:
    #                     continue
    #                 yield split_chunk
    
    def encode(self, text: str) -> list[int]:
        result = []
        if len(self.special_tokens) == 0:
                chunk_bytes = text.encode('utf-8')
                if chunk_bytes in self.encoding_vocab:
                    result.append(self.encoding_vocab[chunk_bytes])
                else:
                    for pretoken_match in re.finditer(self.PAT, text):
                        pretoken = pretoken_match.group()
                        pretoken_bytes = pretoken.encode('utf-8')
                        if pretoken_bytes in self.encoding_vocab:
                            result.append(self.encoding_vocab[pretoken_bytes])
                        else:
                            result.extend(self._encode_pretoken(pretoken_bytes))
        else:
            for chunk in re.split('|'.join(['('+re.escape(st)+')' for st in self.special_tokens]), text):
                if chunk is None:
                    continue
                chunk_bytes = chunk.encode('utf-8')
                if chunk_bytes in self.encoding_vocab:
                    result.append(self.encoding_vocab[chunk_bytes])
                else:
                    for pretoken_match in re.finditer(self.PAT, chunk):
                        pretoken = pretoken_match.group()
                        pretoken_bytes = pretoken.encode('utf-8')
                        if pretoken_bytes in self.encoding_vocab:
                            result.append(self.encoding_vocab[pretoken_bytes])
                        else:
                            result.extend(self._encode_pretoken(pretoken_bytes))
        return result

    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        pretoken_candidate_list = [bytes([b]) for b in pretoken]
        next_candidate_list = []
        while True:
            next_merge = None
            min_merge_index = len(self.merges) + 1  # larger than any possible
            for index in range(len(pretoken_candidate_list)-1):
                current_merge = (pretoken_candidate_list[index], pretoken_candidate_list[index+1])
                if current_merge in self.merges_dict and self.merges_dict[current_merge] < min_merge_index:
                    next_merge = current_merge
                    min_merge_index = self.merges_dict[current_merge]
            index = 0
            while index < len(pretoken_candidate_list):
                if index == len(pretoken_candidate_list)-1:
                    next_candidate_list.append(pretoken_candidate_list[index])
                    break
                current_merge = (pretoken_candidate_list[index], pretoken_candidate_list[index+1])
                if current_merge == next_merge:
                    next_candidate_list.append(b''.join(current_merge))
                    index += 1
                else:
                    next_candidate_list.append(pretoken_candidate_list[index])
                index += 1
            # no change after a loop
            if len(pretoken_candidate_list) == len(next_candidate_list):
                break
            else:
                pretoken_candidate_list = next_candidate_list
                next_candidate_list = []
        return [self.encoding_vocab[token] for token in pretoken_candidate_list]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for iter_str in iterable:
            yield from self.encode(iter_str)

    def decode(self, ids: list[int]) ->str:
        result_bytes = b''.join([self.vocab[id] for id in ids])
        return result_bytes.decode('utf-8', errors='replace')