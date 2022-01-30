#pylint: disable=unused-import, import-error
''' Contains file-hashing-related stuff
'''

import hashlib
import zlib
from pathlib import Path
from lib.global_var import get_global_variables

GV = get_global_variables()

DEFAULT_VALUE = {
    zlib.adler32: 1,
    zlib.crc32: 0
}

def file_digest( _file: Path ) -> int:
    ''' Returns _file's digest
    code based on maxschlepzig's answer
      https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
    '''
    hash_algorithm, block_size = GV.CFG['Hash_algorithm'], GV.CFG['Block_size']
    b  = bytearray(block_size)
    mv = memoryview(b)
    try:
        # hashlib version
        h = hash_algorithm()
        with _file.open('rb', buffering=0) as f:
            for n in iter(lambda : f.readinto(mv), 0):
                h.update(mv[:n])
        # convert bytes digest to int
        return int.from_bytes(h.digest(), byteorder='big')
    except TypeError:
        # zlib version
        digest: int = DEFAULT_VALUE[hash_algorithm]
        with _file.open('rb', buffering=0) as f:
            for n in iter(lambda : f.readinto(mv), 0):
                digest = hash_algorithm( mv[:n], digest ) # data, value
        return digest
