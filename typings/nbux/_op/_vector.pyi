

import aopt.utils.numba as nbu
import numpy as np

@nbu.jti
def dot(x: np.ndarray, y: np.ndarray) -> float:
    
    ...

@nbu.jti
def ndot(x: np.ndarray, y: np.ndarray) -> float:
    
    ...

@nbu.jti
def doti(x: np.ndarray) -> float:
    
    ...

@nbu.jti
def tridot(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> (float, float):
    
    ...

@nbu.jti
def l2nm(x: np.ndarray) -> float:
    
    ...

@nbu.jti
def cxny(dst: np.ndarray, src: np.ndarray):
    
    ...

@nbu.jti
def nx(dst: np.ndarray):
    
    ...

@nbu.jti
def cxpy(dst: np.ndarray, v1: float, src: np.ndarray):
    
    ...

@nbu.jti
def cxay(dst: np.ndarray, v1: float, src):
    
    ...

@nbu.jti
def cxapy(dst: np.ndarray, v1: float, v2: float, src):
    
    ...

@nbu.jti
def axpy(dst: np.ndarray, v: float, src: np.ndarray):
    
    ...

@nbu.jti
def axay(dst: np.ndarray, v1: float, src: np.ndarray):
    
    ...

@nbu.jti
def axapy(dst: np.ndarray, v1: float, v2: float, src: np.ndarray):
    
    ...

@nbu.jti
def pxaxpy(dst: np.ndarray, v1: float, v2: float, src: np.ndarray):
    
    ...

@nbu.jti
def pxaxy(dst: np.ndarray, v1: float, src: np.ndarray):
    
    ...

@nbu.jti
def cxapypz(dst: np.ndarray, v1: float, v2: float, src1: np.ndarray, src2: np.ndarray):
    
    ...

@nbu.jti
def axapypz(dst: np.ndarray, v1: float, v2: float, src1: np.ndarray, src2: np.ndarray):
    
    ...

@nbu.jti
def cxapyz(dst: np.ndarray, v1: float, src1: np.ndarray, src2: np.ndarray):
    
    ...

@nbu.jti
def cxayz(dst: np.ndarray, src1: np.ndarray, src2: np.ndarray):
    
    ...

@nbu.jti
def cxypz(dst1: np.ndarray, dst2: np.ndarray, a: float, b: float, src: np.ndarray): # -> tuple[Any, Any]:
    
    ...

@nbu.jti
def axypz(dst1: np.ndarray, dst2: np.ndarray, a: float, b: float, src: np.ndarray): # -> tuple[Any, Any]:
    
    ...

@nbu.jti
def vmax(x: np.ndarray) -> float:
    
    ...

@nbu.jti
def vmin(x: np.ndarray) -> float:
    
    ...

@nbu.jti
def vminmax(x: np.ndarray) -> (float, float):
    
    ...

@nbu.jti
def argminmax(x: np.ndarray) -> (float, float):
    
    ...

@nbu.jtic
def dtrace(x):
    
    ...

@nbu.jtic
def dadd(x, v): # -> None:
    
    ...

@nbu.jtic
def dvadd(x, v): # -> None:
    
    ...

@nbu.jtic
def dmult(x, v): # -> None:
    
    ...

@nbu.jtic
def dvmult(x, v): # -> None:
    
    ...

