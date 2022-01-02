#pylint: disable=unused-import
''' A very simple script to test
'''

import hashlib
import zlib
import timeit
from pathlib import Path

hashlib_algorithms = [ 
    'md5',
    'sha1',
    'sha224',
    'sha256',
    'sha384',
    'sha512',
    'blake2b',
    'blake2s',
    'sha3_224',
    'sha3_256',
    'sha3_384',
    'sha3_512'
]

zlib_algorithms = [
    'adler32',
    'crc32'
]

def test_algorithm_hashlib( algo: str, plaintext: str ) -> float:
    ''' Returns the amount of time [s] it takes an algorithm to produce
    a digest 100'000 times given a plaintext.
    '''
    cmd = f"hashlib.{algo}( {plaintext} ).hexdigest()"
    return timeit.timeit(cmd, f"import hashlib", number=100_000 )

def test_algorithm_zlib( algo: str, plaintext: str ) -> float:
    ''' Returns the amount of time [s] it takes an algorithm to produce
    a digest 100'000 times given a plaintext.
    '''
    cmd = f"zlib.{algo}( {plaintext} )"
    return timeit.timeit(cmd, f"import zlib", number=100_000 )


def test_algorithms() -> None:
    ''' Tests all algorithms by measuring performance and displaying results
    '''
    plaintext = Path(__file__).read_bytes()


    performance_data = {
        algo: test_algorithm_hashlib( algo, plaintext )
        for algo in hashlib_algorithms
    }

    performance_data.update({
        algo: test_algorithm_zlib( algo, plaintext )
        for algo in zlib_algorithms
    })

    all_algorithms = list(hashlib_algorithms) + list(zlib_algorithms)
    algorithms_by_performance = sorted( all_algorithms, key=lambda x: performance_data[x] )

    for algo in algorithms_by_performance:
        print(f"algo={algo:10s}: time={performance_data[algo] * 1000:.1f} ms")
    

if __name__=='__main__':
    test_algorithms()
    print("END OF PROGRAM")
