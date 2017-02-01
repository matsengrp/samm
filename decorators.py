import time

def measure_time(func):
    """
    A decorator to measure time spent in a function
    Usage:

    @measure_time
    def foo():
        ....

    """
    def func_wrapper(*args, **kwargs):
        st = time.time()
        res = func(*args, **kwargs)
        print "time %s: %f" % (func.__name__, time.time() - st)
        return res
    return func_wrapper
