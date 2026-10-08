"""
Shared settings for the mean-field scripts.

The Wolfram kernel path can be set with the environment variable WOLFRAM_KERNEL;
otherwise the default Linux install path of Wolfram Engine 14.3 is used.
All JSON outputs are written to (and read from) the top-level `results/` folder.
"""

import os

KERNEL = os.environ.get(
    'WOLFRAM_KERNEL',
    '/usr/local/Wolfram/WolframEngine/14.3/Executables/WolframKernel',
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def results_path(name):
    """Absolute path of a file in results/."""
    return os.path.normpath(os.path.join(RESULTS_DIR, name))
