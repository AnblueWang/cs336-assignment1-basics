from typing import Iterator
from pretokenization import PreTokenization
from cs336_basics.pretokenization_example import find_chunk_boundaries
from tokenizer import Tokenizer
import regex as re, os, numpy as np, time

def encode_file(file_path: str, output_path: str, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]) -> Iterator[int]:
    tokenizer = Tokenizer.from_files(Tokenizer,vocab_filepath=vocab_filepath, merges_filepath=merges_filepath, special_tokens=special_tokens)
    
    start = time.perf_counter()
    with open(file_path, 'rb') as file:
        file_boundries = find_chunk_boundaries(file=file, desired_num_chunks=10000, split_special_token=b"<|endoftext|>")
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        chunk_iterator = (file.read(end-start).decode("utf-8", errors="ignore") for start, end in zip(file_boundries[:-1], file_boundries[1:]))
        tokenid_iterator = tokenizer.encode_iterable(chunk_iterator)
        result = np.array(list(tokenid_iterator),dtype=np.int16)
        end = time.perf_counter()
        print(f"Elapsed time is {end-start} seconds for file size {file_size} bytes")
        np.save(output_path, result)
        # file.seek(0, os.SEEK_END)
        # file_size = file.tell()
        # print(f"compression ratio is {file_size/len(result)}")


if __name__ == "__main__":
    # file_path = "./data/owt_train.txt"
    # # file_path = "./data/pure_test.txt"
    # vocab_size = 32000
    # special_tokens = [b'<|endoftext|>']
    # pre_tokenization = PreTokenization(file_path, vocab_size, special_tokens)
    
    # vocab, merges = pre_tokenization.run()
    # print("Longest Vocab:", sorted(vocab.values(), key=lambda x: len(x), reverse=True)[:20])
    # with open(f"vocab_merges_output_{vocab_size}.txt", "w", encoding="utf-8") as f:
    #     f.write(repr(vocab)  + "\n")
    #     f.write(repr(merges) + "\n")
    vocab_filepath = "./vocab_output_10000.txt"
    merges_filepath = "./merges_output_10000.txt"
    owt_vocab_filepath = "./vocab_output_32000.txt"
    owt_merges_filepath = "./merges_output_32000.txt"
    # tokenizer = Tokenizer.from_files(Tokenizer,vocab_filepath=vocab_filepath, merges_filepath=merges_filepath, special_tokens=["<|endoftext|>"])
    # input_str = 'Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>'
    # special_tokens=sorted(["<|endoftext|>", "<|endoftext|><|endoftext|>"], reverse=True)
    # pattern = '|'.join(['('+re.escape(st)+')' for st in special_tokens])
    # print(pattern)
    # print([x for x in re.split(pattern, input_str)])
    # encode_list = tokenizer.encode(input_str)
    # print(encode_list)
    # print([tokenizer.decode([x]) for x in encode_list])
    tiny_sample = "./data/owt_train.txt"
    output_path = "./data/owt-train-tokenid.txt"
    # owt_sample = "./data/owt_samples.txt"
    encode_file(file_path=tiny_sample, output_path=output_path, vocab_filepath=vocab_filepath, merges_filepath=merges_filepath, special_tokens=["<|endoftext|>"])
    # encode_file(file_path=owt_sample, vocab_filepath=vocab_filepath, merges_filepath=merges_filepath, special_tokens=["<|endoftext|>"])
    # encode_file(file_path=tiny_sample, vocab_filepath=owt_vocab_filepath, merges_filepath=owt_merges_filepath, special_tokens=["<|endoftext|>"])
    # encode_file(file_path=owt_sample, vocab_filepath=owt_vocab_filepath, merges_filepath=owt_merges_filepath, special_tokens=["<|endoftext|>"])

