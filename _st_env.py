
didset=False

def pre_start():
    global didset
    if not didset:
        # import os
        # import psutil
        # os.environ["NUMBA_NUM_THREADS"] = "8"
        # os.environ["NUMBA_THREADING_LAYER"] = "tbb"
        # p = psutil.Process()
        # p.cpu_affinity(list(range(8)))
        didset=True
