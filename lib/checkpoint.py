#pylint: disable=import-error
''' Implements file-based temporary data checkpoint system
'''

import pickle
from typing import Any
from lib.global_var import get_global_variables

GV = get_global_variables()

class CheckPoint:
    ''' Saves partial progress to file, makes it easier to continue from
    interrupted execution
    '''
    
    def __init__( self, file_name: str ) -> None:
        if GV.CFG['read_only']:
            self.file = None 
            return
        _save_location = GV.SCRIPT_DIR / 'tmp'
        if not _save_location.is_dir():
            _save_location.mkdir()
        self.file = _save_location / file_name

    def load_progress( self, default: Any = None ) -> Any:
        ''' Loads progress from checkpoint file '''
        if self.file is None or not self.file.is_file():
            return default
        return pickle.loads(self.file.read_bytes())

    def save_progress( self, data: Any ) -> None:
        ''' Saves progress to checkpoint file '''
        if self.file is None:
            return
        GV.LOG.info("Saving progress to %s", self.file)
        self.file.write_bytes( pickle.dumps(data) )

    def remove( self ) -> None:
        ''' Deletes checkpoint file '''
        if self.file is None:
            return
        if self.file.is_file():
            GV.LOG.info("Removing checkpoint %s", self.file.name)
            self.file.unlink()
