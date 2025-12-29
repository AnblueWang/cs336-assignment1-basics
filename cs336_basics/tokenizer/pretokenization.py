import os, regex as re, gc
from typing import BinaryIO
import queue, time, threading
import concurrent.futures
from cs336_basics.pretokenization_example import find_chunk_boundaries

class PreTokenization:
    input_path = ""
    vocab_size = 10000
    special_tokens: list[bytes] = [] 
    vocab: dict[int, bytes] = {} 
    merges: list[tuple[bytes, bytes]] = []
    max_chunk_size = 1024*1024
    chunk_queue = queue.Queue(100)
    read_finished = False
    vocab_lock = threading.Lock()
    pretoken_dict: dict[bytes, tuple[int, list[bytes]]] = {}
    transition_merges_dict: dict[tuple[bytes, bytes], tuple[int, dict[bytes,int]]] = {}

    def __init__(self, input_path:str, vocab_size: int, special_tokens: list[bytes]) -> None:
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.read_finished = False
        self.vocab = {}
        self.merges = []
        self.pretoken_dict = {}
        self.transition_merges_dict = {}
        # initialize vocab with special tokens and byte tokens
        for i, token in enumerate(special_tokens):
            self.vocab[i] = token
        special_len = len(special_tokens)
        for i in range(256):
            self.vocab[i+special_len] = bytes([i])

    def run(self)-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        num_cpu = os.cpu_count()
        with open(self.input_path, "rb") as file:
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            desired_chunk_num = max(file_size//self.max_chunk_size, num_cpu)
            chunk_binaries = find_chunk_boundaries(file, desired_chunk_num, self.special_tokens[0])
            # read chunks
            read_thread = threading.Thread(target=self._read_chunk, args=(file, chunk_binaries))
            read_thread.start()

            pending_tasks = set()
            # consume chunks to split into pretokens and calculate binary merge count
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpu//2) as executor:
                while self.read_finished != True or not self.chunk_queue.empty():
                    if self.chunk_queue.empty():
                        print("split waiting for reading chunks...")
                        time.sleep(1)
                    else:
                        chunk = self.chunk_queue.get()
                        task = executor.submit(self._split_chunk, chunk)
                        task.add_done_callback(self._merge_pretoken_dict)
                        pending_tasks.add(task)
                        if len(pending_tasks) >= num_cpu*4:
                            print("waiting for split tasks to finish...")
                            done, not_done = concurrent.futures.wait(
                                pending_tasks, return_when=concurrent.futures.FIRST_COMPLETED)
                            pending_tasks = not_done
        
        self._get_first_merges()
        self._merge_token()
        return (self.vocab, self.merges)
    
    def _get_next_merge(self) -> tuple[tuple[bytes, bytes], int]:
        # get the top1 merge from transition_merges_dict
        top_merge = None
        top_count = -1
        for merge_key, (merge_count, _) in self.transition_merges_dict.items():
            if merge_count > top_count:
                top_count = merge_count
                top_merge = merge_key
            elif merge_count == top_count:
                if merge_key > top_merge:
                    top_merge = merge_key
        return top_merge, top_count
    
    def _merge_token(self) -> None:
        while len(self.vocab) < self.vocab_size and len(self.transition_merges_dict) > 0:
            top_merge, merge_count = self._get_next_merge()
            print(f"Merge the top1 pair {top_merge} with {merge_count} count. current vocab count {len(self.vocab)} and current merges dict count {len(self.transition_merges_dict)}.")
            self.merges.append(top_merge)
            new_token = b"".join(top_merge)
            token_index = len(self.vocab)
            self.vocab[token_index] = new_token
            _, candidate_pretoken_dict = self.transition_merges_dict[top_merge]
            # print(f"Candidate pretoken bytes set: {candidate_pretoken_set}")
            for pretoken_bytes in candidate_pretoken_dict.keys():
                pretoken_count, pretoken_token_list = self.pretoken_dict[pretoken_bytes]
                new_pretoken_token_list = []
                new_token_index = []
                # merge all new token
                i = 0
                affected_merge_dict: dict[tuple[bytes, bytes], int] = {}
                while i < len(pretoken_token_list):
                    if (i < len(pretoken_token_list)-1) and (pretoken_token_list[i], pretoken_token_list[i+1]) == top_merge:
                        new_token_index.append(len(new_pretoken_token_list))
                        new_pretoken_token_list.append(new_token)
                        # need to reduce the affected merges count based on pretoken_count merges frequency
                        if i-1 >=0:
                            if len(new_pretoken_token_list) >=2 and new_pretoken_token_list[-2] == new_token:
                                None
                            else:
                                affected_merge = (pretoken_token_list[i-1], pretoken_token_list[i])
                                if affected_merge in affected_merge_dict:
                                    affected_merge_dict[affected_merge] += pretoken_count
                                else:
                                    affected_merge_dict[affected_merge] = pretoken_count
                        if i+2 < len(pretoken_token_list):
                            affected_merge = (pretoken_token_list[i+1], pretoken_token_list[i+2])
                            if affected_merge in affected_merge_dict:
                                affected_merge_dict[affected_merge] += pretoken_count
                            else:
                                affected_merge_dict[affected_merge] = pretoken_count
                        i += 1
                    else:
                        new_pretoken_token_list.append(pretoken_token_list[i])
                    i += 1
                # update pretoken_dict pretoken_token_list
                self.pretoken_dict[pretoken_bytes] = (pretoken_count, new_pretoken_token_list)
                # new merges with new token
                for new_token_i in new_token_index:
                    # new token and left token
                    if new_token_i > 0:
                        new_merge = (new_pretoken_token_list[new_token_i-1], new_token)
                        if new_merge in self.transition_merges_dict:
                            new_merge_count, new_pretoken_candidate_dict = self.transition_merges_dict[new_merge]
                            new_merge_count += pretoken_count
                            if pretoken_bytes in new_pretoken_candidate_dict:
                                new_pretoken_candidate_dict[pretoken_bytes] += pretoken_count
                            else:
                                new_pretoken_candidate_dict[pretoken_bytes] = pretoken_count
                            self.transition_merges_dict[new_merge] = (new_merge_count, new_pretoken_candidate_dict)
                        else:
                            self.transition_merges_dict[new_merge] = (pretoken_count, {pretoken_bytes: pretoken_count})
                    # new token and right token
                    if new_token_i < len(new_pretoken_token_list)-1:
                        new_merge = (new_token, new_pretoken_token_list[new_token_i+1])
                        if new_pretoken_token_list[new_token_i+1] == new_token:
                            continue
                        if new_merge in self.transition_merges_dict:
                            new_merge_count, new_pretoken_candidate_dict = self.transition_merges_dict[new_merge]
                            new_merge_count += pretoken_count
                            if pretoken_bytes in new_pretoken_candidate_dict:
                                new_pretoken_candidate_dict[pretoken_bytes] += pretoken_count
                            else:
                                new_pretoken_candidate_dict[pretoken_bytes] = pretoken_count
                            self.transition_merges_dict[new_merge] = (new_merge_count, new_pretoken_candidate_dict)
                        else:
                            self.transition_merges_dict[new_merge] = (pretoken_count, {pretoken_bytes: pretoken_count})
                # reduce the affected merges count
                for affected_merge, reduce_count in affected_merge_dict.items():
                    affected_merge_count, affected_candidate_dict = self.transition_merges_dict[affected_merge]
                    affected_merge_count -= reduce_count
                    if affected_merge_count <= 0:
                        self.transition_merges_dict.pop(affected_merge)
                    else:
                        affected_candidate_dict[pretoken_bytes] -= reduce_count
                        if affected_candidate_dict[pretoken_bytes] <= 0:
                            affected_candidate_dict.pop(pretoken_bytes)
                        self.transition_merges_dict[affected_merge] = (affected_merge_count, affected_candidate_dict)

            self.transition_merges_dict.pop(top_merge)

    def _get_first_merges(self) -> None:
        for pretoken_bytes, (pretoken_count, pretoken_byte_list) in self.pretoken_dict.items():
            for index in range(len(pretoken_byte_list)-1):
                merges_key = (pretoken_byte_list[index], pretoken_byte_list[index+1])
                if merges_key in self.transition_merges_dict:
                    merges_count, candidate_pretoken_dict = self.transition_merges_dict[merges_key]
                    if pretoken_bytes in candidate_pretoken_dict:
                        candidate_pretoken_dict[pretoken_bytes] += pretoken_count
                    else:
                        candidate_pretoken_dict[pretoken_bytes] = pretoken_count
                    self.transition_merges_dict[merges_key] = (merges_count+pretoken_count, candidate_pretoken_dict)
                else:
                    self.transition_merges_dict[merges_key] = (pretoken_count, {pretoken_bytes: pretoken_count})
        print(f"self.transition_merges_dict: {len(self.transition_merges_dict)}")

    
    def _merge_pretoken_dict(self, task: threading.Thread) -> None:
        temp_pretoken_vocab = task.result()
        del task
        print(f"Split task finished one and get {len(temp_pretoken_vocab)} pretokens...")
        with self.vocab_lock:
            for pretoken, pretoken_count in temp_pretoken_vocab.items():
                pretoken_bytes = pretoken.encode('utf-8')
                pretoken_byte_list = [bytes([b]) for b in pretoken_bytes]
                if pretoken_bytes in self.pretoken_dict:
                    cur_pretoken_count, pretoken_byte_list = self.pretoken_dict[pretoken_bytes]
                    self.pretoken_dict[pretoken_bytes] = (cur_pretoken_count + pretoken_count, pretoken_byte_list)
                else:
                    self.pretoken_dict[pretoken_bytes] = (pretoken_count, pretoken_byte_list)
                
                # # add merges and candidate pretokenbytes
                # for index in range(len(pretoken_byte_list)-1):
                #     merges_key = (pretoken_byte_list[index], pretoken_byte_list[index+1])
                #     if merges_key in self.transition_merges_dict:
                #         merges_count, candidate_pretoken_dict = self.transition_merges_dict[merges_key]
                #         if pretoken_bytes in candidate_pretoken_dict:
                #             candidate_pretoken_dict[pretoken_bytes] += pretoken_count
                #         else:
                #             candidate_pretoken_dict[pretoken_bytes] = pretoken_count
                #         self.transition_merges_dict[merges_key] = (merges_count+pretoken_count, candidate_pretoken_dict)
                #     else:
                #         self.transition_merges_dict[merges_key] = (pretoken_count, {pretoken_bytes: pretoken_count})
                
                


        print(f"self.pretoken_dict: {len(self.pretoken_dict)}")
    
    def _read_chunk(self, file: BinaryIO, chunk_binaries) -> None:
        for start, end in zip(chunk_binaries[:-1], chunk_binaries[1:]):
            file.seek(start)
            print(f"Reading file {self.input_path} from position {start}... current chunk queue size {self.chunk_queue.qsize()}")
            chunk = file.read(end - start).decode("utf-8", errors="ignore")
            self.chunk_queue.put(chunk)
            while self.chunk_queue.qsize() > 50:
                time.sleep(10)
                print(f"File reading of {self.input_path} sleep...")
            del chunk
        self.read_finished = True

    def _split_chunk(self, chunk: str) -> dict[str, int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pretoken_dict = {}
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        story_count = 0
        thread_id = threading.get_ident()
        for story in re.split('|'.join([re.escape(x.decode("utf-8", errors="ignore")) for x in self.special_tokens]), chunk):
            story_count += 1
            for pre_token_match in re.finditer(PAT, story):
                pre_token = pre_token_match.group()
                if pre_token not in pretoken_dict:
                    pretoken_dict[pre_token] = 1
                else:
                    pretoken_dict[pre_token] += 1
            if story_count % 50 == 0:
                print(f"Thread {thread_id} has splitten {story_count} stories.")
        return pretoken_dict