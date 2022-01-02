#pylint: disable=invalid-name, line-too-long
''' This script tests different file hashing code and block size on a given
large (>10MB) file and algorithm, to determine optimal parameters for hashing
large files in general on a given system.

Credits for hashing code to ``hashlib`` authors and to authors of answers at:
    https://stackoverflow.com/questions/22058048/hashing-a-file-in-python

Usage guide:
1. Use ``algo_test`` to determine the fastest algorithm on your machine, then
   set it up as HASH_ALGO
2. Get a large file (50MB~500MB) and set up TEST_FILE with a valid path to it
3. Run this script (check PRINT_SORTED for the result representation that you
   prefer) and determine the optimal parameters for best performance.

Conclusion: Now you know the code path/hash algorithm/block size for optimal
performance on your system !
'''

import hashlib, zlib
import timeit
from pathlib import Path
from pprint import pprint

# You may edit these constants
HASH_ALGO = hashlib.sha256 # optimal algorithm; see usage guide step 1
TEST_FILE = Path('./block_test_file.mp4') # your test file path; see usage guide step 2
# False: print results in natural order
# True: print results sorted by most performant first
PRINT_SORTED = True

# Do not touch
TEST_FILE_SIZE_MB = TEST_FILE.stat().st_size / 2**20
DEFAULT_CONFIG = ( 2**16, 'code_path_2' ) # 64kB blocks

KNOWN_HASH = {
    zlib.crc32      : 0x83DE198C,
    hashlib.md5     : 0x33959198550c8aa424da96637ac1781e,
    hashlib.sha1    : 0x0ABED6555E48787ED8F2267DC84D7D9325046AB6,
    hashlib.sha256  : 0x712AC1A13A7BA247666D19E224F812C8A01FB77D7B17C7714EB93DD99E9BAC8D,
    hashlib.sha384  : 0xCBD96068180DDD9DB97173C3DB954EB42DDB2ADE55B4F6182A1D64FECB99CA4C0C8DA06471231E70480FAEFEB7AE498C,
    hashlib.sha512  : 0xbc868c8d07b61071275bf1e9e720fc40c3798f5dd2f2e5307ae3b2b96ee5009f5e96f387a41d622f75cf2b9fb0f5594a43990dd4a12f50d310be10fc63fa2ac8,
    hashlib.sha3_224: 0x49313178a2a2ee917a7eca725cd74f088608909ce158f1ad0ef404cd,
    hashlib.sha3_256: 0x1221e533430b31d0386789f8ade917dd9175b13e0d760304938bf6d3c25b689e,
    hashlib.sha3_512: 0x5d0fce0003494217a973b040d083d1b4737380ebd7adcd65392fad0eef5db41c22ecb312a96b843bb34859b7fa4c31de2a620ad3e1b9624e11e82584c91e7d55
}

def code_path_1( block_size: int ):
    ''' Returns TEST_FILE digest
    code based on Randall Hunt's answer
    '''
    h = HASH_ALGO()
    with TEST_FILE.open('rb') as f:
        while True:
            data = f.read(block_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()

def code_path_2( block_size: int ):
    ''' Returns TEST_FILE digest
    code based on maxschlepzig's answer
    '''
    h = HASH_ALGO()
    b  = bytearray(block_size)
    mv = memoryview(b)
    with TEST_FILE.open('rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()

def test_block( block_size: int, test: str ) -> float:
    ''' Returns the average time [s] (over 5 iterations)
    it takes ``test`` to complete.
    '''
    return timeit.timeit(
        f"{test}({block_size})",
        f"from __main__ import {test}",
        number=5
    ) / 5


def test_blocks() -> None:
    ''' Tests performance of file hashing with different code/block_size parameters.
    Prints the results.
    '''

    sanity_check = {
        f"{test}-2^16": eval(f"{test}( 2**16 )")
        for test in ( 'code_path_1', 'code_path_2' )
    }
    print("Sanity check:")
    pprint(sanity_check)
    if HASH_ALGO in KNOWN_HASH:
        print(f"Expected result: {hex(KNOWN_HASH[HASH_ALGO])}")

    performance_data = {
        f"{test}-2^{block_size_power_2} ({2**block_size_power_2})": \
            test_block( 2**block_size_power_2, test )
        for test in ( 'code_path_1', 'code_path_2' )
        for block_size_power_2 in range(10,30,2)
    }

    print("Results:")
    _keys = sorted(
            performance_data.keys(),
            key=lambda x: performance_data[x]
        ) if PRINT_SORTED \
        else list(performance_data.keys())
    first = None
    for k in _keys:
        _perf = performance_data[k]
        s = f"algo={k:28s}: time={_perf * 1000:5.1f} ms ({TEST_FILE_SIZE_MB/_perf:.1f} MB/s)"

        if PRINT_SORTED:
            if first is None:
                first = _perf
            else:
                s += f" (+{(100 * _perf/first)-100:.1f} %)"
        print(s)


if __name__=='__main__':
    assert TEST_FILE.is_file(), f"Could not access file '{TEST_FILE.resolve()}' ! Is `TEST_FILE` correctly set ?"
    assert TEST_FILE_SIZE_MB>50, f"Test file '{TEST_FILE}' is smaller than 50MB, please use a larger one !"
    print(f"Performing tests on file '{TEST_FILE}' of size {TEST_FILE_SIZE_MB:.1f} MB.")
    print("Performance test ongoing. This can take a minute, please wait ..")
    test_blocks()
    print("END OF PROGRAM")
