from collections.abc import Iterable
import itertools

class Tokenizer(object):
    def encode(self, text: str):
        _ = text
        raise NotImplementedError("This method should be implemented in subclasses")
             
    def encode_iterable(self, iterable: Iterable[str]):
        itr = iter([])
        for text in iterable:
            itr = itertools.chain(itr, self.encode(text))
        return itr
    
    @classmethod
    def from_config(cls, config):
        _ = config
        raise NotImplementedError("This method should be implemented in subclasses")
    
    def encode_chunk(self, input_path, start, end):
        """Encode a byte partition without losing characters at UTF-8 boundaries.

        Both boundaries round forward to the next character boundary, so adjacent
        partitions own each character exactly once. BPE ids may still differ from
        encoding the whole file because merges do not cross chunk boundaries.
        """
        if start < 0 or end < start:
            raise ValueError("chunk bounds must satisfy 0 <= start <= end")
        with open(input_path, "rb") as f:
            def boundary(offset):
                f.seek(offset)
                while byte := f.read(1):
                    if byte[0] & 0xC0 != 0x80:
                        f.seek(-1, 1)
                        break
                return f.tell()

            aligned_start = boundary(start)
            aligned_end = boundary(end)
            f.seek(aligned_start)
            chunk = f.read(aligned_end - aligned_start).decode("utf-8")
            return self.encode(chunk)
    
    def decode(self, ids: list[int], skip_special_tokens: bool = False):
        _ = ids
        raise NotImplementedError("This method should be implemented in subclasses")
    
    def decode_batch(self, ids_batch: list[list[int]], skip_special_tokens: bool = False):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in ids_batch]
