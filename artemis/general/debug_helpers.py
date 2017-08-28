import inspect
import logging
from contextlib import contextmanager
import time
from artemis.general.display import IndentPrint

logging.basicConfig()
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)


@contextmanager
def debug_section(name):
    parent_frame = inspect.currentframe().f_back.f_back
    parent_loc = parent_frame.f_code.co_filename
    _LOGGER.info('Launching {name} @ File "{loc}", line {line}'.format(name=name, loc=parent_loc, line=parent_frame.f_lineno))
    start_time = time.time()
    with IndentPrint():
        yield
    end_time = time.time()
    _LOGGER.info('... Finished {} after {:.4g}s'.format(name, end_time-start_time))
