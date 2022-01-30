#pylint: disable=broad-except
''' Basic test suite for Equilybrium behaviour
WARNING: doesn't cover event HashCollision, for it is 
hard to test (requires known hash collision with >100B of data)
'''

import argparse
import json
import random
import string
import subprocess
import sys
import uuid
import zlib
from pathlib import Path

from lib.hashing import file_digest
from lib.global_var import get_global_variables

GV = get_global_variables()
GV.CFG = {
    'Hash_algorithm': zlib.adler32,
    'Block_size': 2**20,
    'Include_files_with_no_extension': False,
    'Min_file_size': 100,
    'Avoid_at_dirs': True
}


TEST_DIR = Path('./test')
OTHER_DIRS = [ Path('./test/@SynologyHiddenFolder'), Path('./test/$RECYCLE.BIN') ]
EQUILYBRIUM_DIR = Path('./data')

TEST_FILES = {
    'untouched': TEST_DIR / 'untouched.txt',
    'no_ext': TEST_DIR / 'no_ext',
    'dupl_src': TEST_DIR / 'dupl_src.tst',
    'dupl_dst': TEST_DIR / 'dupl_dst.tst',
    'corrupt_mod': TEST_DIR / 'corrupt_mod.xxx',
    'corrupt_add': TEST_DIR / 'corrupt_add.utl',
    'to_move': TEST_DIR / 'to_move.lol',
    'to_remove': TEST_DIR / 'to_remove.ext',
    'syno_to_ignore': Path('./test/@SynologyHiddenFolder/syno_to_ignore.ext'),
    'recycle_to_ignore': Path('./test/$RECYCLE.BIN/recycle_to_ignore.ext'),
    'small_file': TEST_DIR / 'small.txt'
}

CONFIG_FILE = Path('./equilybrium.config.ini')
CONFIG_CONTENT = '''[Settings]
ID_method = Python uuid
A = {}
B = 1234567890
Hash_algorithm = zlib.adler32
Block_size = 2 ** 18
Avoid_at_dirs = {}
Min_file_size = {}
Include_files_with_no_extension = {}
Excluded_extensions = 
[Monitored Directories]
A = ["./test"]
'''


def random_string( length: int = 100 ) -> str:
    ''' Generate random string to put in file '''
    allowed_chars = string.ascii_letters + string.punctuation
    return ''.join(random.choice(allowed_chars) for _ in range(length)) 


def generate_hash_collision_adler32( length: int, destination_file_base: Path ) -> dict:
    ''' Bruteforce-style collision generation: generates 2 files of at least `length` size
    (not same size for both) with same adler32 digest.
    Returns the actual paths of files generated
    '''
    allowed_chars = string.ascii_letters + string.punctuation

    def generate_str_and_hash( _len ) -> tuple:
        s = bytes(''.join(random.choice(allowed_chars) for _ in range(_len)), encoding='utf8')
        return s, zlib.adler32(s)

    # Bruteforce
    print("Generating hash collision. This may take some time ..")
    N = 1_000
    collision = None
    h = None
    while collision is None:
        hashes = {}
        for _ in range(N):
            _str, _hash = generate_str_and_hash( length )
            hashes[_hash] = _str
            
        while True:
            _str,_hash = generate_str_and_hash( length+1 )
            if _hash in hashes:
                collision = [ _str, hashes[_hash] ]
                h = _hash
                break

    # Generating two files with hash collision
    print("Found hash collision. Creating files..")
    destination_files = []
    for idx,content in enumerate(collision):
        dest = destination_file_base.with_suffix(f'.{idx}.txt')
        dest.write_bytes(content)
        destination_files.append(dest)

    return {
        'paths': [ x.as_posix() for x in destination_files ],
        'hash': h
    }


def create_test_files() -> None:
    ''' Generate test files '''

    print("Generating test files")

    for _dir in [ TEST_DIR ] + OTHER_DIRS:
        if not _dir.is_dir():
            print(f"mkdir {_dir}")
            _dir.mkdir()

    for title,file_path in TEST_FILES.items():
        if title=='dupl_dst' or file_path.is_file():
            continue
        
        print(f"Writing file {file_path}")
        with file_path.open('w', encoding='utf8') as f:
            f.write( random_string(
                length=GV.CFG['Min_file_size'] - (1 if title=='small_file' else 0)
            ) )

    GV.COLLISION_FILES = generate_hash_collision_adler32(
        length=GV.CFG['Min_file_size'],
        destination_file_base=TEST_DIR / 'collision.txt'
    )
    TEST_FILES['collision1'], TEST_FILES['collision2'] = ( Path(file_path) for file_path in GV.COLLISION_FILES['paths'] )


def edit_files_after_first_pass() -> None:
    ''' Performs modifications to trigger Equilybrium Events
    '''
    # duplicate file
    print(f"Duplicating {TEST_FILES['dupl_src']} to {TEST_FILES['dupl_dst']}")
    TEST_FILES['dupl_dst'].write_bytes(TEST_FILES['dupl_src'].read_bytes())

    # additive modification
    print(f"Editing {TEST_FILES['corrupt_add']}: appending 'a' char")
    with TEST_FILES['corrupt_add'].open('a',encoding='utf8') as f:
        f.write('a')
        
    # in-place modification
    mod_content = original_content = bytearray(TEST_FILES['corrupt_mod'].read_bytes())
    mod_content[0] += 1
    print(f"Editing {TEST_FILES['corrupt_mod']} first char: {original_content[0]} -> {mod_content[0]}")
    TEST_FILES['corrupt_mod'].write_bytes(mod_content)

    # rename/move
    _renamed = TEST_FILES['to_move'].with_name('renamed'+TEST_FILES['to_move'].suffix)
    print(f"Moving {TEST_FILES['to_move']} to {_renamed}")
    TEST_FILES['to_move'].rename(_renamed)

    # deletion
    print(f"Removing {TEST_FILES['to_remove']}")
    TEST_FILES['to_remove'].unlink()


def check_data() -> None:
    ''' Checks data directory contents for proper expected behaviour
    ''' 
    print("Checking test 'data' contents")

    def check_db() -> None:
        _eq_data = json.loads(Path('./data/A.DB.json').read_text(encoding='utf8'))
        
        for file_path in TEST_DIR.glob('**/*' if GV.CFG['Include_files_with_no_extension'] else '**/*.*'):
            _size, _path = file_path.stat().st_size, file_path.as_posix()
            if _size < GV.CFG['Min_file_size']:
                continue
            if GV.CFG['Avoid_at_dirs'] and any(part.startswith('@') for part in file_path.parts[:-1]):
                continue
            if any(part=='$RECYCLE.BIN' for part in file_path.parts[:-1]):
                continue
            
            try:
                _hash = str(file_digest(file_path))
            except OSError:
                continue

            print(f"DB: File {_path} with hash {_hash} and size {_size}")
            for eq_entry in list(_eq_data[_hash]):
                if eq_entry[0]==_size and eq_entry[1]==_path:
                    _eq_data[_hash].remove(eq_entry)
                    break
            else:
                raise ValueError(f"No equivalent found for {_path},{_hash},{_size}")
        
        assert all( not bool(_eq_data[k]) for k in _eq_data )


    def check_NewFile() -> None:
        _eq_data = [ json.loads(line) for line in Path('./data/A.Event.NewFile.log').read_text(encoding='utf8').splitlines() ]
        for file_path in TEST_FILES.values():
            _size, _path = file_path.stat().st_size if file_path.is_file() else GV.CFG['Min_file_size'], file_path.as_posix()
            if _size < GV.CFG['Min_file_size']:
                continue
            if (not GV.CFG['Include_files_with_no_extension']) and file_path.suffix=='':
                continue
            if GV.CFG['Avoid_at_dirs'] and any(part.startswith('@') for part in file_path.parts[:-1]):
                continue
            if any(part=='$RECYCLE.BIN' for part in file_path.parts[:-1]):
                continue
            if file_path.name=='to_remove':
                continue
            
            print(f"NewFile: File {_path} with size {_size}")
            for eq_entry in _eq_data:
                if eq_entry['path'] == _path:
                    _eq_data.remove(eq_entry)
                    break
            else:
                raise ValueError(f"No equivalent found for {_path},{_size}")
        
        assert len(_eq_data)==0, f"_eq_data={_eq_data} not empty !"


    def check_VerificationFailed() -> None:
        _eq_data = [ json.loads(line) for line in Path('./data/A.Event.VerificationFailed.log').read_text(encoding='utf8').splitlines() ]
        print(f"VerificationFailed: _eq_data={_eq_data}")
        for file_path in (TEST_FILES['corrupt_add'], TEST_FILES['corrupt_mod']):
            _path = file_path.as_posix()
            print(f"VerificationFailed: File {_path}")
            for eq_entry in _eq_data:
                if eq_entry['path'] == _path:
                    _eq_data.remove(eq_entry)
                    break
            else:
                raise ValueError(f"No equivalent found for {_path}")
        
        assert len(_eq_data)==0


    def check_FileMovedOrRenamed() -> None:
        _eq_data = [ json.loads(line) for line in Path('./data/A.Event.FileMovedOrRenamed.log').read_text(encoding='utf8').splitlines() ]
        print(f"FileMovedOrRenamed: _eq_data={_eq_data}")
        
        _to_rename, _renamed = TEST_FILES['to_move'], TEST_FILES['to_move'].with_name('renamed'+TEST_FILES['to_move'].suffix)
        
        assert len(_eq_data)==1 and len(_eq_data[0])==2
        assert _eq_data[0]['moved_from']==_to_rename.as_posix() and _eq_data[0]['moved_to']==_renamed.as_posix()


    def check_DuplicateFile() -> None:
        _eq_data = [ json.loads(line) for line in Path('./data/A.Event.DuplicateFile.log').read_text(encoding='utf8').splitlines() ]
        print(f"DuplicateFile: _eq_data={_eq_data}")
        
        _to_rename, _renamed = TEST_FILES['to_move'], TEST_FILES['to_move'].with_name('renamed'+TEST_FILES['to_move'].suffix)
        
        assert len(_eq_data)==1
        _expected_entries = [ TEST_FILES['dupl_src'].as_posix(), TEST_FILES['dupl_dst'].as_posix() ]
        for _entry in list(_eq_data[0]["duplicate_files"]):
            for _ex_entry in _expected_entries:
                if _entry==_ex_entry:
                    _expected_entries.remove(_ex_entry)
                    _eq_data[0]["duplicate_files"].remove(_entry)
        assert _expected_entries==[] and _eq_data[0]["duplicate_files"]==[]


    def check_HashSanityCheck() -> None:
        
        for file_path in GV.COLLISION_FILES['paths']:
            file_p = Path(file_path)
            print(f"HashSanityCheck: {file_path} with hash {file_digest(file_p)} and size {file_p.stat().st_size}")


    def check_HashCollision() -> None:
        _eq_data = [ json.loads(line) for line in Path('./data/A.Event.HashCollision.log').read_text(encoding='utf8').splitlines() ]
        print(f"HashCollision: {len(_eq_data)} item(s)")

        assert len(_eq_data)==1
        assert all( _eq_data[0][y] in GV.COLLISION_FILES['paths'] for y in ('reference','other_file') )
        assert _eq_data[0]['hash']==GV.COLLISION_FILES['hash']

    tests = {
        'DB': check_db,
        'NewFile': check_NewFile,
        'VerificationFailed': check_VerificationFailed,
        'FileMovedOrRenamed': check_FileMovedOrRenamed,
        'DuplicateFile': check_DuplicateFile,
        'HashSanityCheck': check_HashSanityCheck,
        'HashCollision': check_HashCollision,
    }
    for title,func in tests.items():
        try:
            func()
        except Exception:
            print(f"Error raised while checking {title}!")
            raise

    print("ALL TESTS FINISHED WITH NO ERROR")


def remove_all() -> None:
    ''' Cleanup of generated files/folders '''

    # remove test files
    for _path in TEST_DIR.rglob('**/*'):
        if _path.is_file():
            print(f"Removing {_path}")
            _path.unlink()
    
    # remove test dirs
    for _path in OTHER_DIRS:
        if _path.is_dir():
            print(f"Removing {_path}")
            _path.rmdir()
    if TEST_DIR.is_dir():
        print(f"Removing {TEST_DIR}")
        TEST_DIR.rmdir()


def cli_args() -> argparse.Namespace:
    ''' Parse CLI arguments, implement help '''
    parser = argparse.ArgumentParser(
        prog='Equilybrium-test'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Removes generated files/folders; reverses changes'
    )
    return parser.parse_args()


def make_test_config() -> None:
    ''' Generate a test-specific config file for Equilybrium '''
    current_uuid = uuid.getnode()
    with CONFIG_FILE.open('w', encoding='utf8') as f:
        f.write( CONFIG_CONTENT.format(
            current_uuid,
            'A' if GV.CFG['Avoid_at_dirs'] else '',
            GV.CFG['Min_file_size'],
            GV.CFG['Include_files_with_no_extension']
        ) )


def run_equilybrium() -> None:
    ''' Runs Equilybrium, properly configured for testing '''
    assert Path('equilybrium-dev.py').is_file(), "equilybrium-dev.py not found!"

    # Temporarily replace config file
    TMP_CONFIG_FILE = CONFIG_FILE.with_suffix('.ini.old')
    assert not TMP_CONFIG_FILE.is_file(), f"{TMP_CONFIG_FILE} should not exist!"
    if CONFIG_FILE.is_file():
        CONFIG_FILE.rename( CONFIG_FILE.with_suffix('.ini.old') )
    make_test_config()

    # Run Equilybrium
    command = [sys.executable, 'equilybrium-dev.py']
    print(f"Running {command}")
    try:
        subprocess.run(command, check=True)
    finally:
        # restore config file
        CONFIG_FILE.unlink()
        if TMP_CONFIG_FILE.is_file():
            TMP_CONFIG_FILE.rename(CONFIG_FILE)


def preserve_data() -> None:
    ''' Moves 'data' folder to avoid overwriting and conflicting data
    '''
    print("Preserving 'data' files")
    TMP = EQUILYBRIUM_DIR.with_name('data_OLD')
    if EQUILYBRIUM_DIR.is_dir():
        assert not TMP.is_dir()
        EQUILYBRIUM_DIR.rename(TMP)


def restore_data() -> None:
    ''' Moves 'data' folder to avoid overwriting and conflicting data
    '''

    print("Restoring preserved 'data' files")

    # remove test files/dir
    if EQUILYBRIUM_DIR.is_dir():
        for _path in EQUILYBRIUM_DIR.rglob('**/*'):
            if _path.is_file():
                print(f"Removing {_path}")
                _path.unlink()
        EQUILYBRIUM_DIR.rmdir()

    TMP = EQUILYBRIUM_DIR.with_name('data_OLD')
    if TMP.is_dir():
        print(f"Renaming {TMP} -> {EQUILYBRIUM_DIR}")
        TMP.rename(EQUILYBRIUM_DIR)


if __name__=='__main__':

    _args = cli_args()
    if _args.clean:
        remove_all()
        restore_data()
    else:
        preserve_data()
        create_test_files()
        run_equilybrium()
        edit_files_after_first_pass()
        run_equilybrium()
        check_data()
    