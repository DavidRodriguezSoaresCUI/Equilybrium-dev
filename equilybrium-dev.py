#pylint: disable=eval-used, global-statement, unused-import, invalid-name, trailing-whitespace, line-too-long, ungrouped-imports
''' Script for reading and verifying file digests.

On memory usage: 

- At launch equilybrium uses ~30MB of RAM

- in RAM: It takes 350~450 bytes to store 1 entry, so a DB
with 1M entries would have an estimated size around 400MB.

- in JSON file: It takes 140~300 bytes to store 1 entry, so a DB
with 1M entries would have an estimated size around 200MB.

- Optimizing RAM/JSON file sizes: use a hashing algorithm with smaller digest

- RAM memory footprint was optimized quite a bit. Further optimization seems unwarranted
  and would require significant changes to the code and/or to the features of Equilybrium.

'''

import argparse
import configparser
import datetime
import functools
import hashlib
import itertools
import json
import logging
import sys
import uuid
import zlib
from collections import defaultdict, namedtuple
from enum    import Enum
from pathlib import Path
from time    import time
from typing  import List, Tuple

try:
    from pympler.asizeof import asizeof
except ImportError:
    print("Failed to import pympler. You can install it with `pip install -r equilybrium.requirements.txt`. Fallback `sys.getsiseof` used instead.")
    from sys import getsizeof as asizeof

#################### global config variable ####################

# TL;DR: I know. It's okay though, only ``read_config`` may write to it.
CFG = None
LOG_CFG = 'stderr+file'

#################### Some setup ####################

THIS_FILE = Path( __file__ ).resolve()
LOG_FORMAT = "[%(levelname)s:%(funcName)s] %(message)s"
LOG = logging.getLogger( __file__ )
LOGLEVEL = logging.DEBUG
LOG.setLevel(level=LOGLEVEL)

DEFAULT_VALUE = {
    zlib.adler32: 1,
    zlib.crc32: 0
}

DB_entry = namedtuple('DB_entry', field_names=['size','path'])

#################### Helper functions/classes ####################

class Event(Enum):
    ''' Enumerated all event types that need to be logged.
    See reference for explanation for each element.
    '''
    NewFile = 1
    VerificationFailed = 2
    FileMovedOrRenamed = 3
    FileNotFound = 4
    HashCollision = 5
    DuplicateFile = 6
    OtherProblem = 7


@functools.lru_cache(maxsize=None)
def get_DB_file_path() -> Path:
    ''' Returns path to DB file
    Q: Why a cached function and not a global constant ?
    A: Either way is fine.
    '''
    return THIS_FILE.with_suffix( f".{CFG['role']}.DB.json" )


@functools.lru_cache(maxsize=None)
def get_event_file_path( event: Event ) -> Path:
    ''' Returns path to event file
    Q: Why a cached function and not a global constant ?
    A: Either way is fine.
    '''
    return THIS_FILE.with_suffix( f".{CFG['role']}.{event}.log" )


def log_event( event: Event, msg: str ) -> None:
    ''' Logs (appends) message into the correct file given role and event type.
    '''
    if CFG['read_only']:
        LOG.debug("--simulate: Did not log event to file")
        LOG.info("%s: %s", event, msg)
        return
    event_f = get_event_file_path( event )
    with event_f.open( 'a', encoding='utf8' ) as f:
        f.write( msg + '\n' )


def save_DB( _DB: dict ) -> None:
    ''' Save DB object to file for later use
    Note: By default the json library uses the following encoding representations:
    - int dict keys as str
    - namedtuples as list of their entries (names are lost)
    Therefore an additional step is required (load_DB.decode_DB_entries) when loading
    the saved DB.
    '''
    if CFG['read_only']:
        LOG.debug("--simulate: Did not save DB to file")
        return
    _DB_f = get_DB_file_path()
    if _DB_f.is_file():
        LOG.info("Overwriting DB file '%s'", _DB_f)
    else:
        LOG.info("Saving DB file '%s'", _DB_f)
    _DB_f.write_text(json.dumps(_DB, indent=2))


def load_DB() -> dict:
    ''' Load previously generated DB object from file.
    Returns None on DB file not existing
    '''

    def decode_DB_entries( db: dict ):
        ''' Implements conversion:
          { <digest:str>: [ <DB_entry_as_list:list> ] } => { <digest:int>: [ <entry:DB_entry> ] }
        '''
        for k in list(db.keys()):
            db[int(k)] = [ DB_entry( size=entry[0], path=entry[1] ) for entry in db[k] ]
            del db[k]
        return db

    _DB_f = get_DB_file_path()
    if _DB_f.is_file():
        res = decode_DB_entries(json.loads(_DB_f.read_text() ))
        LOG.info("Loading DB from file '%s' (%d elements)", _DB_f, len(res))
        return res

    LOG.info("DB file '%s' doesn't exist.", _DB_f)
    LOG.info("New DB")
    return dict()


def show_DB_file_size( _DB: dict = None) -> None:
    ''' Self-explanatory
    If DB variable is given, print stats of it
    '''

    def humansize( nbytes: int ) -> str:
        ''' code from
          https://stackoverflow.com/questions/14996453/python-libraries-to-calculate-human-readable-filesize-from-bytes
        '''
        suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        i = 0
        while nbytes >= 1024 and i < len(suffixes)-1:
            nbytes /= 1024.
            i += 1
        f = ('%.2f' % nbytes) # .rstrip('0').rstrip('.')
        return '%s %s' % (f, suffixes[i])

    _DB_f = get_DB_file_path()
    if _DB:
        nb_entries = sum( len(v) for v in _DB.values() )
        file_size_bytes = _DB_f.stat().st_size if _DB_f.is_file() else 0
        var_size_bytes = asizeof(_DB)
        LOG.info(
            "Entries in DB: %d\nDB file size: %s (%.1f byte/entry)\nSize in memory: %s (%.1f byte/entry)",
            nb_entries,
            humansize(file_size_bytes),
            0 if file_size_bytes == 0 else file_size_bytes/nb_entries,
            humansize(var_size_bytes),
            var_size_bytes / nb_entries
        )
    else:
        LOG.info(
            "DB size=%d",
            _DB_f.stat().st_size if _DB_f.is_file() else 0
        )


def print_cfg() -> None:
    ''' For debug purposes
    '''
    global CFG
    msg = "Config:\n" + '\n'.join( f"> '{k}': {v} ({type(v)})" for k,v in CFG.items() )
    msg += "\nMonitored directories:\n" + '\n'.join( f"> {_dir}" for _dir in CFG['dirs'] )
    LOG.debug( msg )


def file_count( root: Path ) -> int:
    ''' Returns the number of files in `root` '''
    return sum( 1 for item in root.glob( '**/*' if CFG['include_noext_files'] else '**/*.*' ) if item.is_file() )


def file_collector( root: Path ) -> Tuple[int,Path,int]:
    ''' Easy to use tool to collect files matching a pattern (recursive or not), using pathlib.glob.
    Collect files matching given pattern(s) '''

    idx: int = 0
    for item in root.glob( '**/*' if CFG['include_noext_files'] else '**/*.*'):
        if not item.is_file():
            continue
        idx += 1
        valid_file = (
            ('$RECYCLE.BIN' not in item.parts) # do not collect files in the trash
            and (item.suffix not in CFG['excluded_extensions']) # exclude files with certain extensions
            and ( # conditionnally avoid directories beginning with '@' symbol
                (CFG['avoid_at_dirs'] is False)
                or (not any( p.startswith('@') for p in item.parts[:-1] ))
            )
        )
        if valid_file:
            yield idx, item, item.stat().st_size


class MySpinner:
    ''' My simple spinner, one-function, easy to use and supports text
    on the left of the spinning wheel !
    '''

    def __init__( self ) -> None:
        self.last_txt_len = 0
        self.spinner = itertools.cycle(['-', '\\', '|', '/'])


    def animation( self, text: str = '', no_spinner: bool = False ):
        ''' update animation, with the option to print text on the left of the
        spinner character
        '''
        tmplen = (self.last_txt_len + 1)
        # erase previous message
        # why do backspaces, then whitespace, then backspaces again :
        # because backspace alone didn't consistently erase previous text
        sys.stdout.write( '\b' * tmplen + ' ' * tmplen + '\b' * tmplen ) 

        # write new message, truncated to not overflow terminal width (necessary for best results)
        spinner_symbol = '' if no_spinner else next(self.spinner)
        message = text + spinner_symbol
        sys.stdout.write( 
            message
        )
        sys.stdout.flush()
        self.last_txt_len = len( message ) - (0 if no_spinner else 1)

#################### Hash-related functions ####################

def file_digest( _file: Path ) -> int:
    ''' Returns _file's digest
    code based on maxschlepzig's answer
      https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
    '''
    hash_algorithm = CFG['Hash_algorithm']
    b  = bytearray(CFG['Block_size'])
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


def hash_dir( dir_to_hash: Path, reference_DB: dict, animate: bool = True ) -> dict:
    ''' Given a directory, hash all files in it recursively,
    then return a dictionnary:
      { 
          <file_digest:int>:
              <(size=<file_size:int>, path=<file_path:str>):DB_entry>
      }
    Note: file_relative_path is a posix-style relative path rooted at
    user-selected directory (non-inclusive)
    '''
    
    # File collection and hashing
    _DB = defaultdict(list)
    total_size_bytes: int = 0 # count processed bytes
    processed_files : int = 0 # count processed files
    LOG.info("Processing directory '%s' ..", dir_to_hash)
    nb_files = file_count( dir_to_hash )

    if nb_files==0:
        LOG.info("No file to process")
        return _DB
        
    spinner = MySpinner()
    start_t = progress_t = time()
    for idx,_file, file_size in file_collector( dir_to_hash ):
        # stdout progress animation
        if animate:
            spinner.animation( f"[{idx+1}/{nb_files}] ({total_size_bytes / max((time()-start_t) * 2**20,1):.1f} MiB/s) " )

        # log progress to file every 5 seconds
        if 'file' in LOG_CFG and time() - progress_t > 5:
            THIS_FILE.with_suffix(f".{CFG['role']}.progress").write_text(
                f"{dir_to_hash} [{idx+1}/{nb_files}] ({total_size_bytes / max((time()-start_t) * 2**20,1):.1f} MiB/s)",
                encoding='utf8'
            )
            progress_t = time()

        # Filter out files that are too small
        if file_size < CFG['min_file_size']:
            continue
        
        # compute file digest
        try:
            _hash = file_digest(_file)
        except PermissionError:
            LOG.warning("Could not access '%s' !", _file)
            continue
        
        # update stats and build entry
        processed_files += 1
        total_size_bytes += file_size
        entry = DB_entry(
            size=file_size,
            path=_file.relative_to(dir_to_hash).as_posix()
        )
        
        # We need to check for hash collisions in the partial DB before adding
        # a new entry
        if not verify_against_partial( _DB, _hash, entry, dir_to_hash):
            # there was an issue with ``entry``
            continue
        
        # In case of successful verification, the corresponding entry is to be
        # removed from reference DB
        entry_to_remove = verify_against_reference( reference_DB, _hash, entry )
        if entry_to_remove is not None:
            LOG.debug("Verification ok => removing an entry at %s from reference DB", _hash)
            reference_DB[_hash].remove( entry_to_remove )
            # remove entry in reference_DB when the last element reference_DB[_hash] is removed 
            if not reference_DB[_hash]:
                del reference_DB[_hash]

        # finally add entry
        _DB[_hash].append(entry)

    if animate:
        # add end of line
        print("")
    
    # Performance logging
    elapsed_t = time() - start_t
    LOG.info(
        "Hashed %.1f MiB in %.1f s : %.1f MiB/s",
        total_size_bytes / 2**20,
        elapsed_t,
        total_size_bytes / (elapsed_t * 2**20)
    )
    LOG.info( "Skipped %d files.", nb_files-processed_files )

    return _DB

#################### Settings-related functions ####################

def cli_args() -> argparse.Namespace:
    ''' Parses CLI arguments using argparse
    '''
    parser = argparse.ArgumentParser(
        prog='Equilybrium (dev)',
        description=__doc__
    )
    special_modes = parser.add_mutually_exclusive_group()
    special_modes.add_argument(
        '-s', '--simulate',
        action='store_true',
        help='Run as read only; do not generate files'
    )
    special_modes.add_argument(
        '--write_report',
        action='store_true',
        help='Only write report'
    )
    parser.add_argument(
        '--no_log_file',
        action='store_true',
        help='Redirects LOG messages to file'
    )
    return parser.parse_args()


def set_LOG_handlers( _file: Path = None ) -> None:
    ''' Sets destination(s) for LOG messages
    required: <LOG_CFG:str>; recogsizes: existence of substrings 'file' and 'stderr'
    examples: 'stderr', 'file+stderr', 'geoig8943file0934utg'
    '''

    for _handler in list(LOG.handlers):
        LOG.removeHandler( _handler )

    _formatter = logging.Formatter( LOG_FORMAT )

    # create file handler for logger.
    if 'file' in LOG_CFG:
        role = CFG['role'] if CFG is not None else ''
        fh = logging.FileHandler(
            filename=THIS_FILE.with_suffix(f'.{role}.log') if _file is None else _file,
            mode='a',
            encoding='utf8'
        )
        fh.setLevel(level=LOGLEVEL)
        fh.setFormatter( _formatter )
        # add handlers to logger.
        LOG.addHandler(fh)

    # create console handler for logger.
    if 'stderr' in LOG_CFG:
        # Bug countermeasure: do not add StreamHandler if one already exist
        sh = logging.StreamHandler()
        sh.setLevel(level=logging.DEBUG)
        sh.setFormatter( _formatter )
        LOG.addHandler(sh)


def read_config() -> dict:
    ''' Reads config from equilibrium.ini
    '''
    global CFG

    # open file and read config
    config_file = THIS_FILE.parent / 'equilybrium.config.ini'
    assert config_file.is_file(), f"Could not find file '{config_file}' !"
    config = configparser.ConfigParser()		
    config.read( config_file, encoding='utf8' )

    # Config to be returned
    CFG = dict()

    # Read 'Settings' section
    _settings = config['Settings']
    # A/B differentiation step
    # Checking ID method (only 'Python uuid' planned for now)
    assert _settings.get('ID_method') == 'Python uuid'
    # get uuid
    current_uuid = uuid.getnode()
    for role in ('A','B'):
        if current_uuid == _settings.getint( role ):
            CFG['role'] = role
            set_LOG_handlers() # re-config file handler
            break
    else:
        LOG.warning("System ID %s doesn't correspond to a role in config %s !", current_uuid, config_file)
        CFG['role'] = 'UNK'

    # get hashing algorithm and block size for hashing
    for arg in ('Hash_algorithm','Block_size'):
        value = _settings.get( arg )
        CFG[arg] = eval( value )

    # determine if directories beginning with '@' should be avoided in this instance
    CFG['avoid_at_dirs'] = CFG['role'] in _settings.get( 'Avoid_at_dirs' )

    # Allows to skip small files
    CFG['min_file_size'] = _settings.getint( 'Min_file_size' )

    # Allows to skip certain extensions
    CFG['excluded_extensions'] = set()
    excl_ext = _settings.get('Excluded_extensions')
    if excl_ext and isinstance(excl_ext, str):
        CFG['excluded_extensions'] = set( 
            item if item[0]=='.' else '.'+item 
            for item in [
                _item.strip().lower()
                for _item in excl_ext.split(',')
            ]
        )

    # Should files with no extension be included ?
    CFG['include_noext_files'] = _settings.getboolean('Include_files_with_no_extension')


    # Read 'Monitored Directories' section
    # Read list of directories corresponding to this instance's role
    monitored_dirs_s = config['Monitored Directories'].get( CFG['role'], raw=True )
    # Conversion: JSON list as str -> List[str] -> List[Path]
    # Note: the ``replace`` operation is to properly escape backslashes present in Windows-style paths
    CFG['dirs'] = [
        Path( _dir )
        for _dir in json.loads(monitored_dirs_s.replace('\\', '\\\\'))
    ] if monitored_dirs_s else []
    # Directory existence verification
    warnings = False
    for _dir in CFG['dirs']:
        if not _dir.is_dir():
            LOG.warning(
                "Directory '%s' listed in equilybrium.ini>[Monitored Directories]>%s was not found !",
                _dir,
                CFG['role']
            )
            warnings = True 
    if warnings is False:
        LOG.info("Loaded %d monitored directory/ies successfully !", len(CFG['dirs']))

    # Generated files setup
    equilybrium_f = Path( __file__ ).resolve()
    CFG['DB_file'] = equilybrium_f.with_suffix( f".{CFG['role']}.DB.json" )

#################### Database analysis, event detection functions ####################

def verify_against_reference( reference_DB: dict, computed_hash: int, computed_item: DB_entry ) -> dict:
    ''' Compares reference_DB with partial computed DB for events:
    - NewFile: element in computed_DB but not in reference_DB
    - FileMovedOrRenamed: element in computed_DB has different path+same size than the one in reference_DB
    - HashCollision: element in computed_DB has different path+different size than the one in reference_DB

    Returns the entry that can be removed from the reference DB (or updated with computed values): <entry>
    if verification is successful (file in reference, exists, may be moved/renamed), None if 
    unsuccessful/unavailable (new file, hash collision).

    Note: this step doesn't catch (by design, another step is needed):
    - VerificationFailed
    - FileNotFound
    '''

    if computed_hash not in reference_DB:
        # NewFile
        log_event(Event.NewFile, str(computed_item.path))
        return None

    # computed_hash is in reference_DB => compare to entries
    # provisioned_action: tuple = ( <executable>, <arguments> )
    provisioned_action:       tuple = None
    provisioned_return_value: bool  = None
    for reference_item in reference_DB[computed_hash]:
        if computed_item.path == reference_item.path:
            # same hash+path => same file
            if computed_item.size == reference_item.size:
                # verification successful
                return reference_item
            
            # Special case: hash collision
            log_event(
                Event.HashCollision,
                f"Hash collision: '{reference_item.path}' vs '{computed_item.path}'"
            )
            provisioned_return_value = None
            continue
            
        
        if computed_item.size == reference_item.size:
            # same hash+size but different path => FileMovedOrRenamed or duplicate
            # Note: we provision a FileMovedOrRenamed because if it is a DuplicateFile,
            # it should be verified in a later iteration
            provisioned_action = (
                log_event,
                (
                    Event.FileMovedOrRenamed,
                    f"'{reference_item.path}' -> '{computed_item.path}'"
                )
            )
            provisioned_return_value = reference_item
            continue
        
        # same hash but different path+size => HashCollision
        log_event(
            Event.HashCollision,
            f"Hash collision: '{reference_item.path}' vs '{computed_item.path}'"
        )
        provisioned_return_value = None

    # If we end here, at least ``provisioned_return_value`` was set
    if provisioned_action:
        # Running provisioned action
        provisioned_action[0]( *provisioned_action[1] )

    return provisioned_return_value


def verify_against_partial( partial_DB: defaultdict, computed_hash: int, computed_item: DB_entry, dir_to_hash: Path ) -> bool:
    ''' Compares partial DB with candidate entry. See truth table for details.

    Returns whether or not the entry should be added to the partial DB.
    '''

    if computed_hash not in partial_DB:
        # "typical" situation => add to partial DB
        return True

    # computed_hash is in partial_DB => compare to entries
    # provisioned_action: tuple = ( <executable>, <arguments> )
    provisioned_action:       tuple = None
    provisioned_return_value: bool  = None
    for reference_item in partial_DB[computed_hash]:
        if computed_item.path == reference_item.path:
            # same hash+path => same file => should never have been processed twice !
            log_event(
                Event.OtherProblem,
                f"File processed twice: {dir_to_hash.as_posix()}/{reference_item.path}; " +
                    f"same size: {computed_item.size == reference_item.size}"
            )
            return False
        
        if computed_item.size == reference_item.size:
            # same hash+size but different path => DuplicateFile
            provisioned_action = (
                log_event,
                (
                    Event.DuplicateFile,
                    f"Duplicate detected in '{dir_to_hash.as_posix()}': " +
                        f"'{computed_item.path}' vs '{reference_item.path}'"
                )
            )
            provisioned_return_value = True
            continue
        
        # same hash but different path+size => hash collision
        provisioned_action = None
        log_event(
            Event.HashCollision,
            f"Hash collision in '{dir_to_hash.as_posix()}': '{reference_item.path}' " +
               f"vs '{computed_item.path}'"
        )
        provisioned_return_value = True

    # If we end here, at least ``provisioned_return_value`` was set
    assert provisioned_return_value is not None
    if provisioned_action:
        # Running provisioned action
        provisioned_action[0]( *provisioned_action[1] )

    return provisioned_return_value


def handle_DB_mismatch( reference_DB: dict, updated_DB: dict ) -> None:
    ''' Compares reference_DB with complete computed DB for events:
    - VerificationFailed: file in reference_DB is in computed DB with different hash
    - FileNotFound: file in reference_DB is not present in computed DB
    - DuplicateFile: file in computed DB is not present in reference_DB, same hash+size
    '''

    # Try to match items from reference DB to computed DB using `path`
    LOG.debug("Entries in reference DB to handle: %d", len(reference_DB))
    for computed_hash, computed_items in updated_DB.items():
        for computed_item in computed_items:
            for reference_hash, reference_items in reference_DB.items():
                for reference_item in reference_items:
                    if computed_item.path == reference_item.path:
                        # VerificationFailed
                        log_event(
                            Event.VerificationFailed,
                            f"Hash mismatch on file '{computed_item.path}': {reference_hash} vs {computed_hash}"
                        )
                        reference_DB[reference_hash].remove(reference_item)
                        break

    # Any remaining item in reference DB must be lost files
    for reference_hash, reference_items in reference_DB.items():
        for reference_item in reference_items:
            log_event(
                Event.FileNotFound,
                f"Cannot find file '{reference_item.path}' with hash {reference_hash} and size {reference_item.size}"
            )

#################### Main ####################

def main() -> None:
    ''' main
    '''
    global LOG_CFG
    import time

    # Setup phase
    flags = cli_args()
    if flags.no_log_file:
        LOG_CFG = 'stderr'
        set_LOG_handlers() # disable logging to log file; stderr only

    read_config()
    CFG['read_only'] = flags.simulate

    # post-setup
    LOG.info("Running equilybrium at %s", datetime.datetime.now())
    print_cfg()
    time.sleep(2)

    # Now equilybrium can actually do some work
    reference_DB = load_DB()
    updated_DB = dict()
    for _dir in CFG['dirs']:
        _dir_DB = hash_dir( _dir, reference_DB )
        updated_DB.update(_dir_DB)

    handle_DB_mismatch( reference_DB, updated_DB )
    save_DB( updated_DB )
    show_DB_file_size( updated_DB )

    # remove temporary file
    progress_f = THIS_FILE.with_suffix(f".{CFG['role']}.progress")
    if progress_f.is_file():
        progress_f.unlink()


if __name__=='__main__':
    set_LOG_handlers() # to catch all messages on systems without visibility on stderr
    main()
    LOG.info("END OF PROGRAM")
