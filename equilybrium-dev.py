#pylint: disable=eval-used, global-statement, unused-import, invalid-name, trailing-whitespace, line-too-long, ungrouped-imports, unsubscriptable-object
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
import itertools
import json
import logging
import pickle
import sys
import threading
import zlib
from collections import defaultdict, namedtuple
from pathlib  import Path
from time     import time, sleep
from typing   import List, Tuple, Any, Iterable

from lib.events import Event
from lib.hashing import file_digest
from lib.utils import current_date, execute_if_not_readonly, cli_args, read_config, format_time_duration
from lib.checkpoint import CheckPoint
from lib.database import DB_entry, load_DB, save_DB, show_DB_file_size, update_database, handle_db_mismatch
from lib.global_var import get_global_variables
from lib.scanner import Scanner

GV = get_global_variables()

#################### global config variable ####################

GV.SECONDS_BETWEEN_CP = 15 # Due to runtime file hashing delay, this is technically a "minimum", not an exact value !

#################### Some global constants ####################

GV.THIS_FILE = Path( __file__ ).resolve()
GV.SCRIPT_DIR = GV.THIS_FILE.parent
GV.LOCK = GV.SCRIPT_DIR / 'AutoYoutubeDL.lock'

#################### Setting up logging ####################

GV.LOG_LEVEL = logging.DEBUG
GV.LOG_FORMAT = "[%(levelname)s:%(funcName)s] %(message)s" if GV.LOG_LEVEL==logging.DEBUG else "[%(levelname)s] %(message)s"
GV.LOG_DIR = GV.SCRIPT_DIR / 'logs'
if not GV.LOG_DIR.is_dir():
    GV.LOG_DIR.mkdir()
logging.basicConfig( 
    level=GV.LOG_LEVEL,
    format=GV.LOG_FORMAT
)
GV.LOG = LOG = logging.getLogger( __file__ )

#################### Helper functions/classes ####################

def print_cfg(only_return_str: bool = False) -> None:
    ''' For debug purposes
    '''
    msg = "Config:\n" + '\n'.join( f"> '{k}': {v} ({type(v)})" for k,v in GV.CFG.items() )
    msg += "\nMonitored directories:\n" + ('\n'.join( f"> {_dir}" for _dir in GV.CFG['dirs'] ) if GV.CFG['dirs'] else '<NOTHING>')
    if only_return_str:
        return msg
    LOG.debug( msg )

#################### Hash-related functions ####################

#################### Settings-related functions ####################



#################### Database analysis, event detection functions ####################

# def verify_against_reference( reference_DB: dict, computed_hash: int, computed_item: DB_entry ) -> dict:
#     ''' Compares reference_DB with partial computed DB for events:
#     - NewFile: element in computed_DB but not in reference_DB
#     - FileMovedOrRenamed: element in computed_DB has different path+same size than the one in reference_DB
#     - HashCollision: element in computed_DB has different path+different size than the one in reference_DB

#     Returns the entry that can be removed from the reference DB (or updated with computed values): <entry>
#     if verification is successful (file in reference, exists, may be moved/renamed), None if 
#     unsuccessful/unavailable (new file, hash collision).

#     Note: this step doesn't catch (by design, another step is needed):
#     - VerificationFailed
#     - FileNotFound
#     '''

#     if computed_hash not in reference_DB:
#         # NewFile
#         Event.log_event(Event.NewFile, str(computed_item.path))
#         return None

#     # computed_hash is in reference_DB => compare to entries
#     # provisioned_action: tuple = ( <executable>, <arguments> )
#     provisioned_action:       tuple = None
#     provisioned_return_value: bool  = None
#     for reference_item in reference_DB[computed_hash]:
#         if computed_item.path == reference_item.path:
#             # same hash+path => same file
#             if computed_item.size == reference_item.size:
#                 # verification successful
#                 return reference_item
            
#             # Special case: hash collision
#             Event.log_event(
#                 Event.HashCollision,
#                 f"Hash collision: '{reference_item.path}' vs '{computed_item.path}'"
#             )
#             provisioned_return_value = None
#             continue
            
        
#         if computed_item.size == reference_item.size:
#             # same hash+size but different path => FileMovedOrRenamed or duplicate
#             # Note: we provision a FileMovedOrRenamed because if it is a DuplicateFile,
#             # it should be verified in a later iteration
#             provisioned_action = (
#                 Event.log_event,
#                 (
#                     Event.FileMovedOrRenamed,
#                     f"'{reference_item.path}' -> '{computed_item.path}'"
#                 )
#             )
#             provisioned_return_value = reference_item
#             continue
        
#         # same hash but different path+size => HashCollision
#         Event.log_event(
#             Event.HashCollision,
#             f"Hash collision: '{reference_item.path}' vs '{computed_item.path}'"
#         )
#         provisioned_return_value = None

#     # If we end here, at least ``provisioned_return_value`` was set
#     if provisioned_action:
#         # Running provisioned action
#         provisioned_action[0]( *provisioned_action[1] )

#     return provisioned_return_value


# def verify_against_partial( partial_DB: defaultdict, computed_hash: int, computed_item: DB_entry, dir_to_hash: Path ) -> bool:
#     ''' Compares partial DB with candidate entry. See truth table for details.

#     Returns whether or not the entry should be added to the partial DB.
#     '''

#     if computed_hash not in partial_DB:
#         # "typical" situation => add to partial DB
#         return True

#     # computed_hash is in partial_DB => compare to entries
#     # provisioned_action: tuple = ( <executable>, <arguments> )
#     provisioned_action:       tuple = None
#     provisioned_return_value: bool  = None
#     for reference_item in partial_DB[computed_hash]:
#         if computed_item.path == reference_item.path:
#             # same hash+path => same file => should never have been processed twice !
#             Event.log_event(
#                 Event.OtherProblem,
#                 f"File processed twice: {dir_to_hash.as_posix()}/{reference_item.path}; " +
#                     f"same size: {computed_item.size == reference_item.size}"
#             )
#             return False
        
#         if computed_item.size == reference_item.size:
#             # same hash+size but different path => DuplicateFile
#             provisioned_action = (
#                 Event.log_event,
#                 (
#                     Event.DuplicateFile,
#                     f"Duplicate detected in '{dir_to_hash.as_posix()}': " +
#                         f"'{computed_item.path}' vs '{reference_item.path}'"
#                 )
#             )
#             provisioned_return_value = True
#             continue
        
#         # same hash but different path+size => hash collision
#         provisioned_action = None
#         Event.log_event(
#             Event.HashCollision,
#             f"Hash collision in '{dir_to_hash.as_posix()}': '{reference_item.path}' " +
#                f"vs '{computed_item.path}'"
#         )
#         provisioned_return_value = True

#     # If we end here, at least ``provisioned_return_value`` was set
#     assert provisioned_return_value is not None
#     if provisioned_action:
#         # Running provisioned action
#         provisioned_action[0]( *provisioned_action[1] )

#     return provisioned_return_value


# def handle_DB_mismatch( reference_DB: dict, updated_DB: dict ) -> None:
#     ''' Compares reference_DB with complete computed DB for events:
#     - VerificationFailed: file in reference_DB is in computed DB with different hash
#     - FileNotFound: file in reference_DB is not present in computed DB
#     - DuplicateFile: file in computed DB is not present in reference_DB, same hash+size
#     '''

#     # Try to match items from reference DB to computed DB using `path`
#     LOG.debug("Entries in reference DB to handle: %d", len(reference_DB))
#     for computed_hash, computed_items in updated_DB.items():
#         for computed_item in computed_items:
#             for reference_hash, reference_items in reference_DB.items():
#                 for reference_item in reference_items:
#                     if computed_item.path == reference_item.path:
#                         # VerificationFailed
#                         Event.log_event(
#                             Event.VerificationFailed,
#                             f"Hash mismatch on file '{computed_item.path}': {reference_hash} vs {computed_hash}"
#                         )
#                         reference_DB[reference_hash].remove(reference_item)
#                         break

#     # Any remaining item in reference DB must be lost files
#     for reference_hash, reference_items in reference_DB.items():
#         for reference_item in reference_items:
#             Event.log_event(
#                 Event.FileNotFound,
#                 f"Cannot find file '{reference_item.path}' with hash {reference_hash} and size {reference_item.size}"
#             )


def generate_report() -> None:
    ''' Generates status report of the state of
    files between A and B. This can only be done
    by A.
    '''
    assert GV.CFG['role']=='A', "Only computer with role 'A' can generate report"

    # Load DBs for comparison
    A_DB, B_DB = load_DB( override_role='A' ), load_DB( override_role='B' )

    # neutralize identical items
    common_file_count = 0
    common_hash_count = 0
    for _hash in A_DB:
        if _hash not in B_DB:
            continue
        common_hash_count += 1
        for _a_entry in list(A_DB[_hash]):
            for _b_entry in B_DB[_hash]:
                # same hash+size => same entry
                if _a_entry.size==_b_entry.size: 
                    A_DB[_hash].remove(_a_entry)
                    B_DB[_hash].remove(_b_entry)
                    common_file_count += 1
                    break
    
    list_exclusives = lambda _db: [ e.path for entries in _db.values() for e in entries ]

    # Generate report with differences
    _report_f = GV.SCRIPT_DIR / f'report_{current_date()}.txt'
    with _report_f.open('w',encoding='utf8') as f:
        f.write('[Equilybrium report]')
        f.write(f"\n\nCommon files: {common_file_count}")
        f.write("\n\nFiles exclusive to A:")
        f.write('\n > ' + '\n > '.join(list_exclusives(A_DB)))
        f.write("\n\nFiles exclusive to B:")
        f.write('\n > ' + '\n > '.join(list_exclusives(B_DB)))


def run_scanners(scanners: List[Scanner], abort_switch: threading.Event) -> dict:
    ''' Handles execution lifecycle of scanners (except checkpoint
    file removal)
    '''
    LOG.info("Starting %d scanners ..", len(scanners))
    for s in scanners:
        s.start()
    LOG.info("Waiting for scanners to complete ..")
    try:
        while any(s.is_alive() for s in scanners):
            sleep(.1)
    except KeyboardInterrupt:
        LOG.warning("KEYBOARD INTERRUPT")
        LOG.info("Waiting for all workers to end properly ..")
        abort_switch.set()
        for s in scanners:
            s.join()
        sys.exit(0)
    LOG.info("Combining results ..")
    res = dict()
    for s in scanners:
        local_result = s.get_result()
        update_database(res, local_result)
        s.cleanup() # removing local _db
    
    return res


def write_scanner_report( scanners: List[Scanner] ) -> None:
    ''' Generate 'end of work' report
    '''
    report_file = GV.SCRIPT_DIR / 'data' / f"{GV.CFG['role']}.report.{current_date()}.log"
    with report_file.open('w', encoding='utf8') as f:
        f.write("[Scanner Report]\n")
        f.write(f"Date: {current_date()}\n")
        f.write(print_cfg(only_return_str=True))
        f.write("\n\nScanner stats:\n")
        for idx,_scanner in enumerate(scanners):
            _tmp = ', '.join( f"{k}:{v}" for k,v in _scanner.stats.items() )
            f.write( f'[{idx}] ' + _tmp + '\n' )
        f.write("\nStats:\n")
        f.write(f"> Elapsed time: {format_time_duration(time() - GV.ScannerStart)}\n")
        total_files_read = sum(_scanner.stats['nb_files_read'] for _scanner in scanners)
        f.write(f"> Total files read: {total_files_read}\n")
        total_files_skipped = sum(_scanner.stats['nb_files_skipped'] for _scanner in scanners)
        f.write(f"> Total files skipped: {total_files_skipped}\n")
        total_MiB_read = sum(_scanner.stats['MiB_read'] for _scanner in scanners)
        f.write(f"> Total MiB read: {total_MiB_read}\n")
        
#################### Main ####################

def main() -> None:
    ''' main
    '''

    # Setup phase
    flags = cli_args()
    GV.CFG = read_config()
    GV.CFG['read_only'] = flags.simulate

    if flags.write_report:
        generate_report()
        return

    # post-setup
    LOG.info("Running equilybrium at %s", current_date())
    print_cfg()

    reference_DB = load_DB()

    def group_by_drive( paths: Iterable[Path] ) -> dict:
        res = defaultdict(list)
        for p in paths:
            res[p.drive].append(p)
        return res

    # Now equilybrium can actually do some work
    GV.ScannerStart = time()
    abort_switch = threading.Event()
    scanners = [
        Scanner(__directories, idx, abort_switch)
        for idx,__directories in enumerate(group_by_drive(GV.CFG['dirs']).values())
    ]
    
    updated_DB = run_scanners(scanners, abort_switch)
    write_scanner_report( scanners )
    
    # cleanup
    handle_db_mismatch( reference_DB, updated_DB )
    save_DB( updated_DB )
    show_DB_file_size( updated_DB )

    # Cleanup checkpoint files
    for s in scanners:
        s.cleanup(remove_cp=True)


if __name__=='__main__':
    main()
    LOG.info("END OF PROGRAM")
