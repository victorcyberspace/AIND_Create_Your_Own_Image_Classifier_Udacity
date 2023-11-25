"""
The workspace_utils.py script is a Python script that provides two functions: active_session() and keep_awake(). 
These functions can be used to keep a Jupyter Workspace session alive for long-running tasks.

The active_session() function takes a delay and an interval as arguments. 
The delay argument specifies how long to wait before sending the first keep-alive request. 
The interval argument specifies how often to send keep-alive requests after that.

The keep_awake() function takes an iterable and a delay and an interval as arguments. 
It iterates over the iterable and yields each item, while also sending keep-alive requests to keep the Jupyter Workspace session alive.
"""

import signal

from contextlib import contextmanager

import requests


DELAY = 90 * 60  # interval time in seconds
MIN_DELAY = MIN_INTERVAL = 90 * 60
KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS = {"Metadata-Flavor":"Google"}


def _request_handler(headers):
    def _handler(signum, frame):
        token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
        requests.request("POST", KEEPALIVE_URL, headers={'Authorization': 'STAR {}'.format(token)})
    return _handler


@contextmanager
def active_session(delay=DELAY, interval=15 * 60):
    """
    Example:

    from workspace_utils import active session

    with active_session():
        # do long-running work here
    """
    token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
    headers = {'Authorization': "STAR {}".format(token)}
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        yield
    except requests.exceptions.RequestException:
        print("Failed to send keep-alive request")
    finally:
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)


def keep_awake(iterable, delay=DELAY, interval=MIN_INTERVAL):
    """
    Example:

    from workspace_utils import keep_awake

    for i in keep_awake(range(5)):
        # do iteration with lots of work here
    """
    with active_session(delay, interval):
        for item in iterable:
            yield item

