"""Helper functions for working with audio files in NumPy."""
import contextlib
from collections import deque
from itertools import chain
from sys import getsizeof, stderr

import numpy as np

from pybinsim.pose import Orientation


def get_nearest_neighbour_key(filter_dict, target):
    # Nearest-neighbour selection in Euclidean space, based on
    # https://github.com/pyfar/pyfar/blob/731353e539abce13f599306bfaa93898bc2f179d/pyfar/classes/coordinates.py#L1521
    #
    # Performance matters here, therefore different parts of the code are
    # benchmarked and the fastest implementation was selected.

    if not filter_dict:  # is empty
        return target

    # get string identifier from custom field of target
    custom_id = target[4]

    # get all available orientations from filter dictionary
    # (they were initialized after loading the filters)
    try:
        orientations_kdtree = filter_dict[f"{custom_id}_orientations_kdtree"]
        orientations_sph = filter_dict[f"{custom_id}_orientations_sph"]
    except KeyError:
        raise ValueError(f"custom={custom_id}")

    # from timeit import timeit
    # def _is_custom_key(item):
    #     return item[0][4] == custom_id
    # globals()["filter_dict"] = filter_dict
    # globals()["custom_id"] = custom_id
    # globals()["_is_custom_key"] = _is_custom_key
    # t1 = timeit(
    #     "dict((key, value) for key, value in filter_dict.items() if key[4] == custom_id)",
    #     globals=globals(),
    # )
    # print(f"timeit 1: {t1 * 1000.0:.1f} ms")
    # t2 = timeit(
    #     "dict(filter(_is_custom_key, filter_dict.items()))",
    #     globals=globals(),
    # )
    # print(f"timeit 2: {t2 * 1000.0:.1f} ms")
    # t3 = timeit(
    #     "dict(filter(lambda items: items[0][4] == custom_id, filter_dict.items()))",
    #     globals=globals(),
    # )
    # print(f"timeit 3: {t3 * 1000.0:.1f} ms")

    # match target source via custom field identifier
    filter_dict = dict(
        filter(lambda items: items[0][4] == custom_id, filter_dict.items())
    )
    if not filter_dict:  # is empty
        raise ValueError(f"custom={custom_id}")

    # get target orientation in Cartesian coordinates
    target_orientation_cart = np.asarray(
        sph2cart(azimuth=target[0].yaw, elevation=target[0].pitch, radius=1.0)
    ).flatten()

    # get index of nearest orientation
    _, orientations_index = orientations_kdtree.query(target_orientation_cart)

    # from timeit import timeit
    #
    # globals()["filter_dict"] = filter_dict
    # globals()["orientations_index"] = orientations_index
    # globals()["orientations_sph"] = orientations_sph
    # globals()["target"] = target
    # t1 = timeit("list(filter_dict.keys())[orientations_index]", globals=globals())
    # print(f"timeit 1: {t1 * 1000.0:.1f} ms")
    # t2 = timeit(
    #     "next(key for index, key in enumerate(filter_dict.keys()) if index == orientations_index)",
    #     globals=globals(),
    # )
    # print(f"timeit 2: {t2 * 1000.0:.1f} ms")
    # t3 = timeit(
    #     "Orientation(*orientations_sph[orientations_index]), target[1:]",
    #     globals=globals(),
    # )
    # print(f"timeit 3: {t3 * 1000.0:.1f} ms")

    # create key based on spherical coordinates of nearest orientation
    return Orientation(*orientations_sph[orientations_index]), *target[1:]


def cart2sph(x, y, z):
    radius = np.sqrt(x**2 + y**2 + z**2)
    z_div_r = np.divide(
        z, radius, out=np.zeros_like(radius, dtype=float), where=radius != 0
    )
    colatitude = np.arccos(z_div_r)
    azimuth = np.mod(np.arctan2(y, x), 2 * np.pi)

    # return azimuth, colatitude, radius
    # all above is taken from
    # https://github.com/pyfar/pyfar/blob/731353e539abce13f599306bfaa93898bc2f179d/pyfar/classes/coordinates.py#L2809

    elevation = np.pi / 2 - colatitude
    # return azimuth and elevation in deg
    return np.rad2deg(azimuth), np.rad2deg(elevation), radius


def sph2cart(azimuth, elevation, radius):
    # get azimuth and elevation in deg
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    colatitude = np.pi / 2 - elevation

    # all below is taken from
    # https://github.com/pyfar/pyfar/blob/731353e539abce13f599306bfaa93898bc2f179d/pyfar/classes/coordinates.py#L2866
    azimuth = np.atleast_1d(azimuth)
    colatitude = np.atleast_1d(colatitude)
    radius = np.atleast_1d(radius)

    r_sin_cola = radius * np.sin(colatitude)
    x = r_sin_cola * np.cos(azimuth)
    y = r_sin_cola * np.sin(azimuth)
    z = radius * np.cos(colatitude)

    x[np.abs(x) < np.finfo(x.dtype).eps] = 0
    y[np.abs(y) < np.finfo(y.dtype).eps] = 0
    z[np.abs(z) < np.finfo(x.dtype).eps] = 0

    return x, y, z


# taken from
# https://github.com/mgeier/python-audio/blob/master/audio-files/utility.py
def pcm2float(sig, dtype="float64"):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in "iu":
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


# taken from
# https://github.com/mgeier/python-audio/blob/master/audio-files/utility.py
def float2pcm(sig, dtype="int16"):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != "f":
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in "iu":
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


# taken from
# https://github.com/mgeier/python-audio/blob/master/audio-files/utility.py
def pcm24to32(data, channels=1, normalize=True):
    """Convert 24-bit PCM data to 32-bit.
    Parameters
    ----------
    data : buffer
        A buffer object where each group of 3 bytes represents one
        little-endian 24-bit value.
    channels : int, optional
        Number of channels, by default 1.
    normalize : bool, optional
        If ``True`` (the default) the additional zero-byte is added as
        least significant byte, effectively multiplying each value by
        256, which leads to the maximum 24-bit value being mapped to the
        maximum 32-bit value.  If ``False``, the zero-byte is added as
        most significant byte and the values are not changed.
    Returns
    -------
    numpy.ndarray
        The content of *data* converted to an *int32* array, where each
        value was padded with zero-bits in the least significant byte
        (``normalize=True``) or in the most significant byte
        (``normalize=False``).
    """
    if len(data) % 3 != 0:
        raise ValueError("Size of data must be a multiple of 3 bytes")

    out = np.zeros(len(data) // 3, dtype="<i4")
    out.shape = -1, channels
    temp = out.view("uint8").reshape(-1, 4)
    if normalize:
        # write to last 3 columns, leave LSB at zero
        columns = slice(1, None)
    else:
        # write to first 3 columns, leave MSB at zero
        columns = slice(None, -1)
    temp[:, columns] = np.frombuffer(data, dtype="uint8").reshape(-1, 3)
    return out


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    """Context manager for temporarily setting NumPy print options.
    See https://stackoverflow.com/questions/2891790/pretty-print-a-numpy-array-without-scientific-notation-and-with-given-precision/2891805#2891805
    """
    original = np.get_printoptions()
    try:
        np.set_printoptions(*args, **kwargs)
        yield
    finally:
        np.set_printoptions(**original)


# taken from https://code.activestate.com/recipes/577504/ as recommended by
# https://docs.python.org/3.5/library/sys.html#sys.getsizeof
def total_size(o, handlers=None, verbose=False):
    """Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """

    if handlers is None:
        handlers = {}

    def dict_handler(d):
        return chain.from_iterable(d.items())

    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    # estimate sizeof object without __sizeof__
    default_size = getsizeof(0)

    def sizeof(_o):
        if id(_o) in seen:  # do not double count the same object
            return 0
        seen.add(id(_o))
        s = getsizeof(_o, default_size)

        if verbose:
            print(s, type(_o), repr(_o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(_o, typ):
                s += sum(map(sizeof, handler(_o)))
                break
        return s

    return sizeof(o)
