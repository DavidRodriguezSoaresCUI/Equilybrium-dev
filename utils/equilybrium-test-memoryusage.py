''' This file contain tests I ran to optimize data representations for
reduced RAM footprint. This code isn't functional anymore and is only preserved
as a reference on data representation decisions.
'''

def tests():
    ''' performance/memory usage benchmarks
    '''
    global CFG

    read_config()

    def digest2str( digest: Union[int, bytes, str] ) -> str:
        ''' Converts digest representation to str
        '''
        if isinstance(digest, bytes):
            digest = int.from_bytes(digest, byteorder='big')
        if isinstance(digest, int):
            return hex(digest)[2:]
        if isinstance(digest, str):
            return digest
        raise ValueError(f"digest of type {type(digest)} unsupported!")
    
    # testing hash representation size
    # Observation: int/bytes representation saves 40-50% memory vs str
    if False:
        for _hash_algo in (zlib.adler32, hashlib.sha256, hashlib.sha3_512):
            print(f"Algorithm: {_hash_algo}")
            CFG['Hash_algorithm'] = _hash_algo
            reference_size = None
            for _code_path in (file_digest, file_digest2):
                print(f"Using code path: {_code_path.__name__}")
                res = _code_path( THIS_FILE )
                _this_size = asizeof(res)
                if reference_size is None:
                    reference_size = _this_size
                print(f"digest={digest2str(res)} (type:{type(res)}, size:{_this_size} ~{100*_this_size/reference_size:.0f}%)")
    
    # testing hash representation size between bytes and int
    # Observation: int representation is consistently 1 byte smaller than bytes
    if False:
        for _hash_algo in (zlib.adler32, hashlib.sha256, hashlib.sha3_512):
            print(f"Algorithm: {_hash_algo}")
            CFG['Hash_algorithm'] = _hash_algo
            digest = file_digest2( THIS_FILE )
            if isinstance(digest, bytes):
                digest_alt = int.from_bytes(digest, byteorder='big')
            elif isinstance(digest, int):
                digest_alt = digest.to_bytes((digest.bit_length() + 7) // 8, 'big')
            else:
                raise RuntimeError()
            print(f"digest={digest} (type:{type(digest)}, size:{asizeof(digest)})")
            print(f"digest_alt={digest_alt} (type:{type(digest_alt)}, size:{asizeof(digest_alt)})")


    # testing path representation size
    # Observation: pathlib.PosixPath representation is an order of magnitude larger than str
    if False:
        reference_size = None
        for _path in (THIS_FILE, THIS_FILE.as_posix()):
            _this_size = asizeof(_path)
            if reference_size is None:
                reference_size = _this_size
            print(f"path={_path} (type:{type(_path)}, size:{_this_size} ~{100*_this_size/reference_size:.0f}%)")


def tests2() -> None:
    ''' test DB entry data representation '''

    # Observation1: typically getsizeof(x) < asizeof(x)
    # Observation2: namedtuple representation has 2-3x smaller memory footprint vs dict

    from sys import getsizeof

    from collections import namedtuple
    DB_entry = namedtuple('DB_entry', field_names=['size','path'])

    _size: int = THIS_FILE.stat().st_size
    _path: str = THIS_FILE.as_posix()
    entry1: dict = {
        'size': _size,
        'path': _path
    }
    entry2 = DB_entry( size=_size, path=_path )
    for _entry in (entry1,entry2):
        print(f"entry={_entry} (type:{type(_entry)}, asizeof:{asizeof(_entry)}, getsizeof:{getsizeof(_entry)})")