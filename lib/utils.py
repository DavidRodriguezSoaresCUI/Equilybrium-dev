#pylint: disable=eval-used, import-error, unused-import
''' Contains uncategorized utils
'''

import argparse
import configparser
import functools
import hashlib
import json
import logging
import sys
import uuid
import zlib
from datetime import datetime
from pathlib import Path
from typing import Callable, Any, Union
from lib.global_var import get_global_variables

GV = get_global_variables()

DEFAULT_CONFIG = '''[Settings]
# System IDs are required for Equilybrium to identify their role and create
# their files.
# `ID method`: for now, only 'Python uuid' is implemented
ID_method = Python uuid
A = 
B = 
# Hashing settings
# `Hash algorithm` must be one of: 'md5', 'sha1', 'sha224', 'sha256', 'sha384',
# 'sha512', 'blake2b', 'blake2s', 'sha3_224', 'sha3_256', 'sha3_384',
# 'sha3_512', 'adler32', 'crc32' (without quotes; suffixed by corresponding
# library 'hashlib.' or 'zlib.')
Hash_algorithm = zlib.adler32
# Can be any integer; only edit this value if you determined it is suboptimal
# on your systems
Block_size = 2 ** 18
# Set to role that should avoid directories beginning with '@' symbol (Synology NAS)
Avoid_at_dirs = 
# Set minimum file size to be processed, in bytes
Min_file_size = 100
# Whether or not to process files with no extension
Include_files_with_no_extension = False
# List excluded extensions. Format: comma-separated values, case insensitive, with or without '.'
Excluded_extensions = 
# List excluded files. Format: comma-separated values, case sensitive (ex: Thumbs.db in Windows)
Excluded_files = 

[Monitored Directories]
# Which directories contains files to monitor, on each system ?
# List here between double quotes the absolute path to these folders
A = [
    ""
    ]
B = [
    ""
    ]
'''

def current_date() -> str:
    ''' Returns string representation of current date.
    Precision: minute
    '''
    return datetime.now().strftime("%Y%m%d_%H%M")

@functools.lru_cache(maxsize=None)
def get_data_path() -> Path:
    ''' Returns base path for DB/event files
    Q: Why a cached function and not a global constant ?
    A: Either way is fine.
    '''
    _dir = GV.SCRIPT_DIR / "data"
    if not (GV.CFG['read_only'] or _dir.is_dir()):
        _dir.mkdir(parents=True)
    return _dir


@functools.lru_cache(maxsize=None)
def get_DB_file_path(override_role: str = None) -> Path:
    ''' Returns path to DB file
    Q: Why a cached function and not a global constant ?
    A: Either way is fine.
    '''
    if override_role:
        return get_data_path() / f"{override_role}.DB.json"
    return get_data_path() / f"{GV.CFG['role']}.DB.json"


def execute_if_not_readonly( message: str ) -> Callable:
    ''' Decorator for functions that should be disabled on ``--simulate``
    '''
    def actual_decorator( user_function: Callable ) -> Callable:
        @functools.wraps( user_function )
        def wrapper( *args, **kwargs ) -> Any:
            nonlocal user_function, message
            
            if 'CFG' in GV and GV.CFG['read_only']:
                GV.LOG.info("--simulate: %s", message)
                return
            
            return user_function( *args, **kwargs )

        return wrapper

    if callable( message ):
        # decorator_with_arguments was run without argument => use 
        # default values for expected arguments or raise error
        user_function = message
        message = f"{user_function.__name__} was not executed"
        return actual_decorator(user_function)

    return actual_decorator


def cli_args() -> argparse.Namespace:
    ''' Parses CLI arguments using argparse
    '''
    parser = argparse.ArgumentParser(
        prog='Equilybrium (dev)',
        description=__doc__
    )
    special_modes = parser.add_mutually_exclusive_group()
    special_modes.add_argument(
        '--simulate',
        action='store_true',
        help='Run as read only; do not generate files'
    )
    special_modes.add_argument(
        '--write_report',
        action='store_true',
        help='Only write report'
    )
    special_modes.add_argument(
        '--differential_scan',
        action='store_true',
        help='Only scan files that are NOT in database. This is useful for a quick "update" when editing configuration for example.'
    )
    return parser.parse_args()


@execute_if_not_readonly(message='No filehandler were added')
def set_LOG_handlers(role_override: str = None) -> None:
    ''' Sets file as secondary output for LOG messages
    '''

    for _handler in list(GV.LOG.handlers):
        if isinstance(_handler, logging.FileHandler):
            GV.LOG.removeHandler( _handler )

    # create file handler for logger.
    if role_override:
        _logfile = GV.LOG_DIR / "{}_{}.log".format(role_override, current_date())
    elif GV.CFG is not None and 'role' in GV.CFG:
        _logfile = GV.LOG_DIR / "{}_{}.log".format(GV.CFG['role'], current_date())
    else:
        raise RuntimeError("No role set")
    fh = logging.FileHandler(
        filename=str(_logfile),
        mode='w',
        encoding='utf8'
    )
    fh.setLevel(level=GV.LOG_LEVEL)
    fh.setFormatter( logging.Formatter( GV.LOG_FORMAT ) )
    # add handlers to logger.
    GV.LOG.addHandler(fh)


def make_config( file_path: Path ) -> None:
    ''' Generate a default config file for the user to complete
    ''' 
    GV.LOG.info("No config file found; generating %s ..", file_path)
    file_path.write_text(DEFAULT_CONFIG, encoding='utf8')
    GV.LOG.info("Please fill config file %s with appropriate values.", file_path)
    GV.LOG.info("Note: current machine UUID is %d", uuid.getnode())


def read_config() -> dict:
    ''' Reads config from equilibrium.ini
    '''

    # open file and read config
    config_file = GV.THIS_FILE.parent / 'equilybrium.config.ini'
    if not config_file.is_file():
        make_config(config_file)
        sys.exit(0)

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
            set_LOG_handlers(role_override=role) # re-config file handler
            break
    else:
        GV.LOG.warning("System ID %s doesn't correspond to a role in config %s !", current_uuid, config_file)
        CFG['role'] = 'UNK'

    # get hashing algorithm and block size for hashing
    for arg in ('Hash_algorithm','Block_size'):
        value = _settings.get( arg )
        CFG[arg] = eval( value )

    # determine if directories beginning with '@' should be avoided in this instance
    CFG['avoid_at_dirs'] = CFG['role'] in _settings.get( 'Avoid_at_dirs' )

    # Allows to skip small files
    CFG['min_file_size'] = _settings.getint( 'Min_file_size' )

    # Allows to skip certain extensions/files
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
    CFG['excluded_files'] = set()
    excl_f = _settings.get('Excluded_files')
    if excl_f and isinstance(excl_f, str):
        CFG['excluded_files'] = set( 
            _item.strip()
            for _item in excl_f.split(',')
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
        if len(_dir)>0
    ] if monitored_dirs_s else []
    # Directory existence verification
    if not CFG['dirs']:
        raise ValueError("'Monitored directories' is empty!")
    warnings = False
    for _dir in CFG['dirs']:
        if not _dir.is_dir():
            GV.LOG.warning(
                "Directory '%s' listed in equilybrium.ini>[Monitored Directories]>%s was not found !",
                _dir,
                CFG['role']
            )
            warnings = True 
    if warnings is False:
        GV.LOG.info("Loaded %d monitored directory/ies successfully !", len(CFG['dirs']))

    # Generated files setup
    equilybrium_f = Path( __file__ ).resolve()
    CFG['DB_file'] = equilybrium_f.with_suffix( f".{CFG['role']}.DB.json" )
    return CFG


def format_time_duration(seconds: Union[int,float]) -> str:
    ''' Format time duration in a human-friendly way
    '''
    assert seconds>=0
    _h, _remainder = divmod(seconds, 3600)
    _m, _s = divmod(_remainder, 60)
    _time = ''.join(f"{int(x):02}{y}" for x,y in zip([_h,_m,_s],['H','M','S']) if x!=0 or y=='S')
    if len(_time)>2:
        return _time.lstrip('0')
    return _time
