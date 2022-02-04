#pylint: disable=import-error
''' Contains DB-related stuff
'''

import json
from collections import namedtuple, defaultdict
from pathlib import Path

try:
    from pympler.asizeof import asizeof
except ImportError:
    print("Failed to import pympler. You can install it with `pip install -r equilybrium.requirements.txt`. Fallback `sys.getsiseof` used instead.")
    from sys import getsizeof as asizeof

from lib.events import Event
from lib.global_var import get_global_variables
from lib.utils import execute_if_not_readonly, get_DB_file_path

GV = get_global_variables()

DB_entry = namedtuple('DB_entry', field_names=['size','path'])

@execute_if_not_readonly("Did not save DB to file")
def save_DB( _DB: dict ) -> None:
    ''' Save DB object to file for later use
    Note: By default the json library uses the following encoding representations:
    - int dict keys as str
    - namedtuples as list of their entries (names are lost)
    Therefore an additional step is required (load_DB.decode_DB_entries) when loading
    the saved DB.
    '''
    _DB_f = get_DB_file_path()
    if _DB_f.is_file():
        GV.LOG.info("Overwriting DB file '%s'", _DB_f)
    else:
        GV.LOG.info("Saving DB file '%s'", _DB_f)
    _DB_f.write_text(json.dumps(_DB, indent=2))


def load_DB( override_role: str = None ) -> dict:
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

    _DB_f = get_DB_file_path( override_role )
    if _DB_f.is_file():
        res = decode_DB_entries(json.loads(_DB_f.read_text() ))
        GV.LOG.info("Loading DB from file '%s' (%d elements)", _DB_f, len(res))
        return res

    GV.LOG.info("New DB: file '%s' doesn't exist.", _DB_f)
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
        GV.LOG.info(
            "Entries in DB: %d\nDB file size: %s (%.1f byte/entry)\nSize in memory: %s (%.1f byte/entry)",
            nb_entries,
            humansize(file_size_bytes),
            0 if file_size_bytes == 0 else file_size_bytes/nb_entries,
            humansize(var_size_bytes),
            var_size_bytes / nb_entries
        )
    else:
        GV.LOG.info(
            "DB size=%d",
            _DB_f.stat().st_size if _DB_f.is_file() else 0
        )


def diff_database(reference_db: dict, updated_db: dict) -> tuple:
    ''' Returns the difference between the databases as a tuple:
    <(<uniques_to_reference_db:dict>, <uniques_to_updated_db:dict>):tuple>
    '''
    
    same_entry = lambda a,b: a.size==b.size and a.path==b.path
    nb_common_entries = 0

    def difference(a: dict, b: dict, count_entries: bool = True) -> dict:
        ''' From a and b, returns differences <a-b:dict>
        '''
        nonlocal nb_common_entries
        res = defaultdict(list)
        for _hash in list(a.keys()):
            if _hash in b:
                # compare entries individually
                for a_entry in list(a[_hash]):
                    if not any(same_entry(a_entry,b_entry) for b_entry in b[_hash]):
                        res[_hash].append(a_entry)
                    elif count_entries:
                        nb_common_entries += 1
            else:
                res[_hash] = a[_hash]
                        
        return res

    ref_uniques = difference(reference_db, updated_db)
    upd_uniques = difference(updated_db, reference_db, count_entries=False)
    # entry_counter = lambda x: sum(1 for entries in x.values() for entry in entries)
    # nb_ref_only_entries = entry_counter(ref_uniques)
    # nb_upd_only_entries = entry_counter(upd_uniques)
    # GV.LOG.info("common:%d, reference only:%d, updated only:%d", nb_common_entries, nb_ref_only_entries, nb_upd_only_entries)

    return ref_uniques, upd_uniques #, nb_ref_only_entries+nb_upd_only_entries


def update_database(base_db: dict, partial_db: dict) -> None:
    ''' Adds items of ``partial_db`` in ``base_db``
    '''
    for __hash,__entries in partial_db.items():
        if __hash in base_db:
            for __entry in __entries:
                base_db[__hash].append(__entry)
            continue
        base_db[__hash] = __entries


def handle_db_mismatch(reference_db: dict, updated_db: dict) -> None:
    ''' Handles databasse mismatch, triggers events
    '''
    GV.LOG.info("Analysing database, registering events ..")

    ref_uniques, upd_uniques = diff_database(reference_db, updated_db)
    #GV.LOG.info("Mismatched entries to handle: %d", mismatch_count)

    # Same hash => FileMovedOrRenamed (same size, different path) or HashCollision (different size)
    for ref_hash in list(ref_uniques.keys()):
        for ref_entry in list(ref_uniques[ref_hash]):
            for upd_entry in list(upd_uniques[ref_hash]):
                if upd_entry.size == ref_entry.size:
                    Event.log_event(
                        Event.FileMovedOrRenamed,
                        {'moved_to': upd_entry.path, 'moved_from': ref_entry.path}
                    )
                    upd_uniques[ref_hash].remove(upd_entry)
                    ref_uniques[ref_hash].remove(ref_entry)

    # same hash and size, different paths => DuplicateFile
    duplicate_files = lambda a,b: a.path!=b.path and a.size==b.size
    for _hash, upd_entries in upd_uniques.items():
        for upd_entry in list(upd_entries):
            _dupes = [
                ref_entry.path
                for ref_entry in reference_db.get(_hash,[])
                if duplicate_files(upd_entry,ref_entry)
            ]
            if _dupes:
                Event.log_event(
                    Event.DuplicateFile,
                    {'duplicate_files': [upd_entry.path] + _dupes}
                )
                # upd_uniques[_hash].remove(upd_entry)

    # Different hash, same path => VerificationFailed
    # Try to match items from ref_uniques to upd_uniques using `path`
    _not_new_file = set()
    for upd_hash in list(upd_uniques.keys()):
        for upd_entry in list(upd_uniques[upd_hash]):
            for ref_hash in list(ref_uniques.keys()):
                for ref_entry in list(ref_uniques[ref_hash]):
                    if upd_entry.path == ref_entry.path:
                        Event.log_event(
                            Event.VerificationFailed,
                            {'path': upd_entry.path, 'reference_hash': ref_hash, 'updated_hash': upd_hash}
                        )
                        ref_uniques[ref_hash].remove(ref_entry)
                        _not_new_file.add(upd_entry)
                        break

    # Check new entries for possible hash collision
    # same hash, different size/path => HashCollision
    hash_collision = lambda a,b: a.path!=b.path and a.size!=b.size
    reported_hash_collision = set()
    for upd_hash in upd_uniques:
        for upd_entry in upd_uniques[upd_hash]:
            _other_entries = list(upd_uniques[upd_hash]) + ( list(reference_db[upd_hash]) if upd_hash in reference_db else [] )
            _dupes = [
                _other_entry
                for _other_entry in _other_entries
                if hash_collision(upd_entry,_other_entry) and (_other_entry not in reported_hash_collision)
            ]
            if _dupes:
                reported_hash_collision.add(upd_entry)
            for _ref in _dupes:
                Event.log_event(
                    Event.HashCollision,
                    {'other_file': upd_entry.path, 'reference': _ref.path, 'hash': upd_hash}
                )

    # Anything remaining in ref_uniques is FileNotFound
    for ref_hash in ref_uniques:
        for _entry in ref_uniques[ref_hash]:
            _entry_path = Path(_entry.path)
            if _entry_path.suffix in GV.CFG['excluded_extensions'] or _entry_path.name in GV.CFG['excluded_files']:
                # Excluding previously-allowed extension/file names should not trigger a FileNotFound event
                continue
            Event.log_event(
                Event.FileNotFound,
                {'path': _entry.path}
            )
    
    # Anything remaining in upd_uniques is NewFile
    for upd_hash in upd_uniques:
        for _entry in upd_uniques[upd_hash]:
            if _entry in _not_new_file:
                continue
            Event.log_event(
                Event.NewFile,
                {'path': _entry.path}
            )


def reverse_database(_db: dict) -> dict:
    ''' Takes a { <hash:int> : <entry:DB_entry> } and returns
    a { <posix_path:str>: <hash:int> } reversed database
    '''
    return {
        entry.path: _hash 
        for _hash, entries in _db.items()
        for entry in entries
    }
            