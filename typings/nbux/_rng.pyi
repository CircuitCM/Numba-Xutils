

import nbux._utils as nbu

@nbu.jt
def durstenfeld_p_shuffle(a, k=...): # -> None:
    
    ...

@nbu.jtic
def jt_uniform_rng(a, b): # -> float:
    ...

def set_seed(seed): # -> None:
    ...

_N = ...
scaled_rademacher_rng = ...
@nbu.rgic
def normal_rng_protect(mu=..., sig=..., pr=...): # -> float:
    ...

@nbu.jtic
def place_gauss_s(a, mu=..., sig=...): # -> None:
    ...

@nbu.jtpc
def place_gauss_pl(a, mu=..., sig=...): # -> None:
    ...

@nbu.ir_force_separate_pl(place_gauss_s, place_gauss_pl)
def place_gauss(a, mu=..., sigma=..., parallel=...): # -> None:
    ...

@nbu.rgc
def random_orthogonals(a, ortho_mem, parallel=...): # -> None:
    ...

@nbu.jtic
def place_uniform_s(a, l=..., u=...): # -> None:
    ...

@nbu.jtpc
def place_uniform_pl(a, l=..., u=...): # -> None:
    ...

@nbu.ir_force_separate_pl(place_uniform_s, place_uniform_pl)
def place_uniform(a, l=..., u=..., parallel=...): # -> None:
    ...

@nbu.jtic
def place_gauss_no0_s(a, mu=..., sig=...): # -> None:
    ...

@nbu.jtpc
def place_gauss_no0_pl(a, mu=..., sig=...): # -> None:
    ...

@nbu.ir_force_separate_pl(place_gauss_no0_s, place_gauss_no0_pl)
def place_gauss_no0(a, mu=..., sigma=..., parallel=...): # -> None:
    ...

@nbu.jtic
def place_rademacher_s(a, l=..., u=...): # -> None:
    ...

@nbu.jtpc
def place_rademacher_pl(a, l=..., u=...): # -> None:
    ...

@nbu.ir_force_separate_pl(place_rademacher_s, place_rademacher_pl)
def place_rademacher(a, l=..., u=..., parallel=...): # -> None:
    ...

