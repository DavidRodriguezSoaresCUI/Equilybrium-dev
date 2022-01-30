#pylint: disable=import-error
''' Contains the Scanner class, a Thread subclass that
hashes arbitrary directory contents
'''

import hashlib
import threading
from collections import defaultdict
from time import time
from typing import List, Tuple
from pathlib import Path


from lib.checkpoint import CheckPoint
from lib.database import DB_entry, update_database
from lib.global_var import get_global_variables
from lib.hashing import file_digest

GV = get_global_variables()

def file_count( root: Path ) -> int:
    ''' Returns the number of files in `root` (recursive search) '''
    return sum( 
        1
        for item in root.glob( 
            '**/*' if GV.CFG['include_noext_files'] else '**/*.*'
        )
        if item.is_file()
    )


def file_collector( root: Path ) -> Tuple[int,Path,int]:
    ''' Easy to use tool to collect files matching a pattern (recursive or not), using pathlib.glob.
    Collect files matching given pattern(s) '''

    idx: int = 0
    for item in root.glob( '**/*' if GV.CFG['include_noext_files'] else '**/*.*'):
        if not item.is_file():
            continue
        idx += 1
        valid_file = (
            ('$RECYCLE.BIN' not in item.parts) # do not collect files in the trash
            and (item.suffix not in GV.CFG['excluded_extensions']) # exclude files with certain extensions
            and ( # conditionnally avoid directories beginning with '@' symbol
                (GV.CFG['avoid_at_dirs'] is False)
                or (not any( p.startswith('@') for p in item.parts[:-1] ))
            )
        )
        if valid_file:
            yield idx, item, item.stat().st_size


class Scanner(threading.Thread):
    ''' threading.Thread subclass; hashes arbitrary directory contents '''

    def __init__(self, directories: List[Path], n: int, kill_thread: threading.Event) -> None:
        # calling parent class constructor
        threading.Thread.__init__(self)
        # setting up instance variables
        self._directories = directories
        self._db = dict()
        self._n = n
        _id_str = ''.join(sorted(x.as_posix() for x in directories))
        self._id = hashlib.md5(_id_str.encode('utf8')).hexdigest()
        assert isinstance(self._id, str)
        self._cp = CheckPoint(file_name=f"{self._id}.tmp")
        self._done = False
        self._kill_thread = kill_thread


    def hash_dir(self, dir_to_hash: Path) -> dict:
        ''' Processes a directory: hashes all of its contents '''

        GV.LOG.info("Processing directory '%s' ..", dir_to_hash)
        nb_files = file_count( dir_to_hash )

        if nb_files==0:
            GV.LOG.info("No file to process")
            return dict()
        
        # Load progress
        local_cp = CheckPoint(file_name=f"{self._id}_{dir_to_hash.name}.tmp")
        local_db, _hashed_files, cp_dir = local_cp.load_progress( default=(defaultdict(list),set(),dir_to_hash) )
        if _hashed_files and cp_dir==dir_to_hash:
            GV.LOG.info("Continuing from checkpoint: %d files processed!", len(list(_hashed_files)))

        # stats for execution loop
        total_size_bytes: int = 0 # count processed bytes
        processed_files : int = 0 # count processed files
        min_file_size, interval_between_cp = GV.CFG['min_file_size'], GV.SECONDS_BETWEEN_CP
        start_t = checkpoint_t = time()
        # max used to avoid division by 0 errors
        throughput_MiB = lambda: total_size_bytes / max((time()-start_t) * 2**20,1)

        # process files
        for idx,_file,file_size in file_collector( dir_to_hash ):
            # Forcibly abort execution
            if self._kill_thread.is_set():
                return local_db
            
            # Filter out files that are too small
            if file_size < min_file_size:
                GV.LOG.debug("Skipping '%s': too small (%dB)", _file, file_size)
                continue

            _file_path = _file.as_posix()

            # Skip if already hashed
            if _file_path in _hashed_files:
                GV.LOG.debug("Skipping '%s': already processed", _file_path)
                continue
            
            # Periodically save progress
            _now = time()
            if _now - checkpoint_t > interval_between_cp:
                checkpoint_t = _now
                local_cp.save_progress( (local_db, _hashed_files, dir_to_hash) )
            
            # compute file digest
            GV.LOG.debug("Hashing file %s", _file_path)
            try:
                _hash = file_digest(_file)
            except PermissionError:
                GV.LOG.warning("Could not access '%s' !", _file_path)
                continue
            
            # update stats and build entry
            processed_files += 1
            total_size_bytes += file_size
            entry = DB_entry(
                size=file_size,
                path=_file_path
            )

            # log progress
            GV.LOG.info(
                "%s [%d/%d] (%.1f MiB/s)", 
                dir_to_hash, idx, nb_files, throughput_MiB()
            )

            # finally add entry
            local_db[_hash].append(entry)
            _hashed_files.add(_file_path)
        
        # Performance logging
        elapsed_t = time() - start_t
        GV.LOG.info(
            "Hashed %.1f MiB in %.1f s : %.1f MiB/s",
            total_size_bytes / 2**20,
            elapsed_t,
            throughput_MiB()
        )
        GV.LOG.info( "Processed %d files; Skipped %d files.", processed_files, nb_files-processed_files )

        # cleanup: remove checkpoint at the end
        local_cp.remove()

        return local_db


    def run(self) -> None:
        ''' Given a directory, hash all files in it recursively,
        keeping a dictionnary:
        { 
            <file_digest:int>:
                <[(size=<file_size:int>, path=<file_path:str>)]:List[DB_entry]>
        }
        Note: `file_path` is a posix-style absolute path
        '''
        GV.LOG.info("Started Scanner %d on directories %s", self._n, [str(x) for x in self._directories])

        # load progress
        self._db, _processed_dirs = self._cp.load_progress( default=(dict(),set()) )
        if _processed_dirs:            
            GV.LOG.info("Continuing from checkpoint: %d directories processed!", len(_processed_dirs))
        
        # process directories
        for _dir in self._directories:
            # skip if already processed
            if _dir in _processed_dirs:
                GV.LOG.info("Skipped dir %s: already processed!", _dir.as_posix())
                continue
            # Do work
            partial_db = self.hash_dir(_dir)
            
            # Forcibly abort execution
            if self._kill_thread.is_set():
                GV.LOG.info("Scanner %d execution is being aborted", self._n)
                break

            update_database(self._db, partial_db)
            # save progress
            _processed_dirs.add(_dir)
            self._cp.save_progress( (self._db, _processed_dirs) )

        self._done = True
        GV.LOG.info("Scanner %d finished execution.", self._n)


    def get_result(self) -> dict:  
        ''' get result database '''
        if self._done is not True:
            raise RuntimeError("Error: called Scanner.get_result before it is done executing!")
        if hasattr(self, '_db') is False:
            raise RuntimeError("Error: called Scanner.get_result after cleanup!")
        return self._db


    def cleanup(self, remove_cp: bool = False) -> None:
        ''' remove database and checkpoint '''
        if hasattr(self, '_db'):
            del self._db
        if remove_cp:
            self._cp.remove()
        