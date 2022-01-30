#pylint: disable=import-error
''' Contains Event-related stuff
'''

import functools
import json
from enum import Enum
from pathlib import Path
from lib.utils import get_data_path, execute_if_not_readonly
from lib.global_var import get_global_variables

GV = get_global_variables()

class Event(Enum):
    ''' Enumerated all event types that need to be logged.
    See reference for explanation for each element.
    '''
    NewFile = 1
    VerificationFailed = 2
    FileMovedOrRenamed = 3
    FileNotFound = 4
    HashCollision = 5 # Very unlikely
    DuplicateFile = 6
    OtherProblem = 7 # Unused

    @classmethod
    @execute_if_not_readonly("Did not log event")
    def log_event( cls, event: 'Event', msg: dict ) -> None:
        ''' Logs (appends) message into the correct file given role and event type.
        '''
        event_f = get_event_file_path( event )
        GV.LOG.debug("New %s", event)
        with event_f.open( 'a', encoding='utf8' ) as f:
            f.write( json.dumps(msg) + '\n' )


@functools.lru_cache(maxsize=None)
def get_event_file_path( event: Event ) -> Path:
    ''' Returns path to event file
    Q: Why a cached function and not a global constant ?
    A: Either way is fine.
    '''
    _role = GV.CFG['role']
    return get_data_path() / f"{_role}.{event}.log"
