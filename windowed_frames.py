from collections import deque
from itertools import chain
import numpy as np

def windowed(seq, n, fillvalue=None, step=1):
    """Return a sliding window of width *n* over the given iterable.

        >>> all_windows = windowed([1, 2, 3, 4, 5], 3)
        >>> list(all_windows)
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    When the window is larger than the iterable, *fillvalue* is used in place
    of missing values::

        >>> list(windowed([1, 2, 3], 4))
        [(1, 2, 3, None)]

    Each window will advance in increments of *step*:

        >>> list(windowed([1, 2, 3, 4, 5, 6], 3, fillvalue='!', step=2))
        [(1, 2, 3), (3, 4, 5), (5, 6, '!')]

    To slide into the iterable's items, use :func:`chain` to add filler items
    to the left:

        >>> iterable = [1, 2, 3, 4]
        >>> n = 3
        >>> padding = [None] * (n - 1)
        >>> list(windowed(chain(padding, iterable), 3))
        [(None, None, 1), (None, 1, 2), (1, 2, 3), (2, 3, 4)]

    """
    retlist = []
    if n < 0:
        raise ValueError('n must be >= 0')
    if n == 0:
        # yield tuple()
        # return
        return retlist
    if step < 1:
        raise ValueError('step must be >= 1')

    it = iter(seq)
    window = deque([], n)
    append = window.append

    # Initial deque fill
    for _ in range(n):
        append(next(it, fillvalue))
    # yield tuple(window)
    retlist.append(np.array(window))

    # Appending new items to the right causes old items to fall off the left
    i = 0
    for item in it:
        append(item)
        i = (i + 1) % step
        if i % step == 0:
            # yield tuple(window)
            retlist.append(np.array(window))

    # If there are items from the iterable in the window, pad with the given
    # value and emit them.
    if (i % step) and (step - i < n):
        for _ in range(step - i):
            append(fillvalue)
        # yield tuple(window)
        retlist.append(np.array(window))
    return np.array(retlist)