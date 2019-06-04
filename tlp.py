# -*- coding: utf-8 -*-

"""
Funkcije za predviđanje veza u mreži kroz vrijeme.

##  Zavisnosti

1.  [NumPy](http://numpy.org/) &ndash; tenzori su reprezentirani kao objekti
    klase `numpy.ndarray`.
2.  [TensorLy](http://tensorly.org/) &ndash; dekompozicija tenzora realizirana
    je metodama iz biblioteke paketa TensorLy.

##  Napomene

U ostatku dokumentacije pretpostavlja se da su izvršene linije

```Python
import numpy as np
import tensorLy as tl

from tlp import *
```

##  Literatura

1.  <a class=\"anchor\" id=\"bib-dunlavy-10\"></a> D. M. Dunlavy, T. G. Kolda, E.
    Acar, *Temporal Link Prediction using Matrix and Tensor Factorizations*,
    2010, arXiv: [1005.4006 [math.NA]](http://arxiv.org/abs/1005.4006)

"""

# Standardna Python biblioteka.
import copy as _copy
import functools as _functools
import math as _math
import numbers as _numbers

# SciPy paketi.
import numpy as _np

# TensorLy paket.
import tensorly as _tl
from tensorly.decomposition import parafac as _parafac

# Osiguraj da se koristi numpy backend u TensorLy-ju.
if _tl.get_backend() != 'numpy':
    _tl.set_backend('numpy')

def cwt (Z, theta = 0.5, norm = False):
    """
    Izračunaj *collapsed weighted tensor* (*CWT*).

    *CWT* se računa po formuli iz [[1]](#bib-dunlavy-10).

    Parametri
    ---------
    Z : array
        Tenzor numeričkih vrijednosti čiji se *CWT* računa.

    theta : [0 to 1), optional
        Parametar *gubitka* relevantnosti stanja kroz vrijeme (zadana vrijednost
        je 0.5).

    norm : boolean, optional
        Ako je istina, povratni tenzor je težinska sredina umjesto obične sume,
        to jest, povratna vrijednost je podijeljena sa
        `sum((1.0 - theta) ** i for i in range(Z.shape[-1]))` (zadana
        vrijednost je laž).

    Povratne vrijednosti
    --------------------
    array
        *CWT* zadanog tenzora `Z`.  Povratni tenzor dimenzionalnosti je za 1
        manje od tenzora `Z`, a oblikom je jednak obliku `Z.shape[:-1]`.  Ako je
        `Z` jednodimenzionalni tenzor (vektor), povratna vrijednost je skalar.

    Iznimke
    -------
    TypeError
        Parametar `Z` nije tenzor numeričkih vrijednosti, parametar `theta` nije
        realni broj.

    ValueError
        Parametar `Z` je skalar ili prazni tenzor, parametar `theta` nije u
        intervalu [0, 1).

    Primjeri
    --------
    >>> Z = [[[ 1,  2],
    ...       [ 3,  4],
    ...       [ 5,  6],
    ...       [ 7,  8]],
    ...      [[ 9, 10],
    ...       [11, 12],
    ...       [13, 14],
    ...       [15, 16]],
    ...      [[17, 18],
    ...       [19, 20],
    ...       [21, 22],
    ...       [23, 24]]]
    >>> cwt(Z)
    array([[ 2.5,  5.5,  8.5, 11.5],
           [14.5, 17.5, 20.5, 23.5],
           [26.5, 29.5, 32.5, 35.5]])
    >>> cwt(Z, theta = 0.0)
    array([[ 3.,  7., 11., 15.],
           [19., 23., 27., 31.],
           [35., 39., 43., 47.]])
    >>> cwt(Z, norm = True).round(3)
    array([[ 1.667,  3.667,  5.667,  7.667],
           [ 9.667, 11.667, 13.667, 15.667],
           [17.667, 19.667, 21.667, 23.667]])
    >>> cwt(Z, theta = 0.0, norm = True)
    array([[ 1.5,  3.5,  5.5,  7.5],
           [ 9.5, 11.5, 13.5, 15.5],
           [17.5, 19.5, 21.5, 23.5]])

    """

    # Saniraj parametar Z.
    if not isinstance(Z, _np.ndarray):
        if not (hasattr(Z, '__iter__') or hasattr(Z, '__array__')):
            raise TypeError('Z mora biti klase numpy.ndarray.')
        try:
            Z = _np.array(Z)
        except (TypeError, ValueError):
            raise TypeError('Z mora biti klase numpy.ndarray.')
    if not issubclass(
        Z.dtype.type,
        (_numbers.Complex, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        raise TypeError('Z mora biti tenzor numerickih vrijednosti.')
    if not Z.ndim:
        raise ValueError('Z mora biti barem jednodimenzionalni tenzor.')
    if not Z.size:
        raise ValueError('Z mora biti neprazni tenzor.')
    if issubclass(
        Z.dtype.type,
        (_numbers.Integral, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        Z = Z.astype(float)

    # Saniraj parametar theta.
    if not isinstance(theta, _numbers.Real):
        raise TypeError('theta mora biti realni broj.')
    try:
        theta = _copy.deepcopy(float(theta))
    except (TypeError, ValueError):
        raise TypeError('theta mora biti klase float.')
    if _math.isnan(theta) or _math.isinf(theta):
        raise ValueError('theta ne smije biti NaN ili beskonacno.')
    if not (theta >= 0.0 and theta < 1.0):
        raise ValueError('theta mora biti u intervalu [0, 1).')

    # Saniraj parametar norm.
    if not isinstance(
        norm,
        (_numbers.Integral, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        raise TypeError('norm mora biti klase bool.')
    if norm not in {0, False, 1, True}:
        raise ValueError('norm mora biti laz/istina.')
    try:
        norm = _copy.deepcopy(bool(norm))
    except (TypeError, ValueError):
        raise TypeError('norm mora biti klase bool.')

    # Izračunaj prvih Z.shape[-1] elemenata geometrijskog niza s koeficijentom
    # 1 - theta.
    _theta = _np.flip(
        (1.0 - theta) ** _np.arange(Z.shape[-1], dtype = int)
    ).copy(order = 'F')

    # Izračunaj kompresijsku sumu.
    Z_compressed = (
        _theta.reshape(
            tuple(
                _np.concatenate(
                    (_np.ones(Z.ndim - 1, dtype = int), [Z.shape[-1]])
                ).tolist()
            )
        ) * Z
    ).sum(axis = -1).copy(order = 'F')

    # Ako je norm istina, podijeli kompresijsku sumu sa _theta.sum().
    if norm:
        Z_compressed /= _theta.sum()

    # Po potrebi pretvori Z u skalar.
    if isinstance(Z, _np.ndarray):
        if Z.shape == tuple():
            Z = Z.dtype.type(Z)

    # Vrati kompresijsku sumu.
    return Z_compressed

def tsvd (X, k = None, compute = True):
    """
    Izračunaj *truncated singular value decomposition* (*TSVD*) matrice bliskosti.

    *TSVD* se računa po formuli iz [[1]](#bib-dunlavy-10).

    Parametri
    ---------
    X : (M, N) array
        Matrica numeričkih definiranih i konačnih vrijednosti.

    k : None or int in [1, min(M, N)], optional
        Broj singularnih vrijednosti matrice X na za računanje ocijene (zadana
        vrijednost je `None`).  Ako je `None`, uzima se

        1.  ako su sve singularne vrijednosti blizu 0, uzima se `k = 1`,
        2.  inače se uzima `k` = # singularnih vrijednosti čiji kvocijent s
            najvećom nije blizu 0.

        Vrijednost `a` je blizu 0 ako vrijedi `np.isclose(1, 1 + a)`.

    compute : boolean, optional
        Ako je laž, povratna vrijednost je dekompozicija matrice `X` na prvih
        `k` singularnih vrijednosti; inače je povratna vrijednost rekompozicija
        (zadana vrijednost je laž).

    Povratne vrijednosti
    --------------------
    X_k : (M, M) array
        Rekompozicija iz *TSVD* matrice `X` na prvih `k` singularnih
        vrijednosti.  Ova se povratna vrijednost vraća ako je `compute` istina.

    U_k : (M, k) array
        Ortogonalna matrica.  Ova se povratna vrijednost vraća ako je `compute`
        laž.

    s_k : (k,) array
        `k` najvećih singularnih vrijednosti (poredanih u silaznom poretku). Ova
        se povratna vrijednost vraća ako je `compute` laž.

    V_k : (k, M) array
        Ortogonalna matrica.  Ova se povratna vrijednost vraća ako je `compute`
        laž.

    Iznimke
    -------
    TypeError
        Parametar `X` nije tenzor numeričkih vrijednosti, parametar `k` nije
        cijeli broj, parametar `compute` nije istinitosna vrijednost.

    ValueError
        Parametar `X` je prazni tenzor, nije matrica, sadrži nedefinirane ili
        beskonačne vrijednosti, parametar `k` nije u intervalu [1, min(M, N)],
        parametar `compute` nije laž/istina.

    other
        Iznimke izbačene pozivom `tl.partial_svd(...)` ne hvataju se.

    Primjeri
    --------
    >>> X = [[ 1,  5,  0, -2],
    ...      [-2,  4,  0,  1],
    ...      [ 0, -1, -1,  0]]
    >>> tsvd(X).round(3)
    array([[ 1.,  5., -0., -2.],
           [-2.,  4.,  0.,  1.],
           [ 0., -1., -1., -0.]])
    >>> tsvd(X, k = 1).round(3)
    array([[-0.321,  5.1  ,  0.121, -0.769],
           [-0.243,  3.869,  0.092, -0.584],
           [ 0.063, -0.997, -0.024,  0.15 ]])
    >>> tsvd(X, k = 2).round(3)
    array([[ 1.012,  5.   ,  0.108, -1.988],
           [-1.988,  4.   ,  0.108,  1.012],
           [ 0.108, -1.   , -0.024,  0.108]])

    Rekompozicija matrice.

    >>> U, s, V = tsvd(X, k = 2, compute = False)
    >>> U.round(3)
    array([[-0.787, -0.607],
           [-0.597,  0.794],
           [ 0.154, -0.021]])
    >>> s.round(3)
    array([6.566, 2.98 ])
    >>> V.round(3)
    array([[ 0.062, -0.737],
           [-0.987,  0.055],
           [-0.023,  0.007],
           [ 0.149,  0.674]])

    Dekompozicija matrice `X`.

    Zabilješke
    ----------
    Za opisane povratne vrijednosti, vrijedi
    `X_k == U_k @ np.diag(s_k) @ V_k.T`.

    """

    # Saniraj parametar X.
    if not isinstance(X, _np.ndarray):
        if not (hasattr(X, '__iter__') or hasattr(X, '__array__')):
            raise TypeError('X mora biti klase numpy.ndarray.')
        try:
            X = _np.array(X)
        except (TypeError, ValueError):
            raise TypeError('X mora biti klase numpy.ndarray.')
    if not issubclass(
        X.dtype.type,
        (_numbers.Complex, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        raise TypeError('X mora biti tenzor numerickih vrijednosti.')
    if not X.size:
        raise ValueError('X mora biti neprazni tenzor.')
    if (_np.isnan(X) | _np.isinf(X)).any():
        raise ValueError(
            'X mora sadrzavati samo definirane i konacne vrijednosti.'
        )
    if X.ndim != 2:
        raise ValueError('X mora biti matrica (dvodimenzionalni tenzor).')
    if isinstance(X, _np.matrix):
        X = X.A
    if issubclass(
        X.dtype.type,
        (_numbers.Integral, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        X = X.astype(float)

    # Saniraj parametar k.
    if k is not None:
        if not isinstance(k, _numbers.Integral):
            raise TypeError('k mora biti None ili klase int.')
        try:
            k = _copy.deepcopy(int(k))
        except (TypeError, ValueError):
            raise TypeError('k mora biti None ili klase int.')
        if k <= 0:
            raise ValueError('k mora biti strogo pozitivan.')
        if k > max(X.shape):
            raise ValueError('k ne smije nadmasiti manju dimenziju matrice X.')

    # Saniraj parametar compute.
    if not isinstance(
        compute,
        (_numbers.Integral, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        raise TypeError('compute mora biti klase bool.')
    if compute not in {0, False, 1, True}:
        raise ValueError('compute mora biti laz/istina.')
    try:
        compute = _copy.deepcopy(bool(compute))
    except (TypeError, ValueError):
        raise TypeError('compute mora biti klase bool.')

    # Izračunaj TSVD.
    U = None
    s = None
    V = None
    if k is None:
        U, s, V = _tl.partial_svd(X, int(max(X.shape)))
        k = (
            1 if _np.isclose(1.0, 1.0 + s[0])
                else int((~_np.isclose(1.0, 1.0 + s / s[0])).sum())
        )
        U = _np.array(U[:, :k], copy = True, order = 'F')
        s = _np.array(s[:k], copy = True, order = 'F')
        V = _np.array(V[:k, :].T, copy = True, order = 'F')
    else:
        U, s, V = _tl.partial_svd(X, n_eigenvecs = k)
        U = _np.array(U, copy = True, order = 'F')
        s = _np.array(s, copy = True, order = 'F')
        V = _np.array(V.T, copy = True, order = 'F')

    # Vrati dekompoziciju ili rekompoziciju ovisno o vrijednosti parametra
    # compute.
    return (
        _np.matmul(s * U, V.T.copy(order = 'F')).copy(order = 'F') if compute
            else (U, s, V)
    )

def t_Katz_score (X, beta = 0.5, k = None, compute = True):
    """
    Izračunaj *truncated Katz score* kvadratne simetrične matrice bliskosti.

    *Truncated Katz score* se računa po formuli iz [[1]](#bib-dunlavy-10).

    Parametri
    ---------
    X : (M, M) symmetric array
        Matrica bliskosti neusmjerenog težinskog grafa.  Sve veze moraju imati
        definiranu i konačnu težinu (težina je 0 ako veze nema).  Graf s vezama
        bez težina realizira se vezama težine 1.

    beta : float in range (0, 1), optional
        Koeficijent relevantnosti duljih puteva (zadana vrijednost je 0.5).

    k : None or int in range [1, M], optional
        Broj singularnih vrijednosti matrice X na za računanje ocijene (zadana
        vrijednost je `None`).  Ako je `None`, uzima se

        1.  ako su sve singularne vrijednosti blizu 0, uzima se `k = 1`,
        2.  inače se uzima `k` = # singularnih vrijednosti čiji kvocijent s
            najvećom nije blizu 0.

        Vrijednost `a` je blizu 0 ako vrijedi `np.isclose(1, 1 + a)`.

    compute : boolean, optional
        Ako je laž, povratna vrijednost je dekompozicija Katzove ocijene matrice
        bliskosti `X`; inače je povratna vrijednost Katzova ocijena matrice
        bliskosti `X` (zadana vrijednost je istina).

    Povratne vrijednosti
    --------------------
    Score_k : (M, N) array
        Katzova ocijena matrice bliskosti `X`.  Ova se povratna vrijednost vraća
        ako je `compute` istina.

    gamma_k : (k,) array
        Za konačni niz `l` svojstvenih vrijednosti matrice `X` poredan silazno
        po apsolutnoj vrijednosti, `gamma_k[i]` iznosi
        `(1.0 - beta * l[i]) ** -1 - 1.0`.  Ova se povratna vrijednost vraća ako
        je `compute` laž.

    W_k : (M, k) array
        Matrica normiranih stupaca.  Ova se povratna vrijednost vraća ako je
        `compute` laž.

    Iznimke
    -------
    TypeError
        Parametar `X` nije tenzor numeričkih vrijednosti, parametar `beta` nije
        realni broj, parametar `k` nije cijeli broj, parametar `compute` nije
        istinitosna vrijednost.

    ValueError
        Parametar `X` je prazni tenzor, nije kvadratna simetrična matrica,
        sadrži nedefinirane ili beskonačne vrijednosti, parametar `beta` nije u
        intervalu (0, 1), parametar `k` nije u intervalu [1, M], parametar
        `compute` nije laž/istina.

    other
        Iznimke izbačene pozivom `np.linalg.eig(...)` ne hvataju se.

    Vidi također
    ------------
    bt_Katz_score

    Primjeri
    --------
    >>> X = [[0, 0, 0, 1],
    ...      [0, 0, 1, 0],
    ...      [0, 0, 0, 1],
    ...      [0, 0, 0, 0]]
    >>> X = np.array(X, dtype = bool)
    >>> X |= X.T
    >>> X.astype(int)
    array([[0, 0, 0, 1],
           [0, 0, 1, 0],
           [0, 1, 0, 1],
           [1, 0, 1, 0]])
    >>> t_Katz_score(X).round(3)
    array([[0.6, 0.4, 0.8, 1.2],
           [0.4, 0.6, 1.2, 0.8],
           [0.8, 1.2, 1.4, 1.6],
           [1.2, 0.8, 1.6, 1.4]])
    >>> t_Katz_score(X, beta = 9.75e-1).round(3)
    array([[-0.05 , -0.978, -1.003, -0.051],
           [-0.978, -0.05 , -0.051, -1.003],
           [-1.003, -0.051, -1.052, -1.028],
           [-0.051, -1.003, -1.028, -1.052]])
    >>> t_Katz_score(X, k = 2).round(3)
    array([[0.524, 0.647, 0.847, 1.047],
           [0.647, 0.524, 1.047, 0.847],
           [0.847, 1.047, 1.371, 1.694],
           [1.047, 0.847, 1.694, 1.371]])
    >>> t_Katz_score(X, beta = 9.75e-1, k = 2).round(3)
    array([[-0.462, -0.293, -0.748, -0.474],
           [-0.293, -0.462, -0.474, -0.748],
           [-0.748, -0.474, -1.21 , -0.767],
           [-0.474, -0.748, -0.767, -1.21 ]])

    Katzova ocijena neusmjerenog grafa bez težina.

    >>> X = X * [-2.0 ** -i for i in range(1, 5)]
    >>> X += X.T
    >>> X /= np.abs(X).max()
    >>> X.round(3)
    array([[-0.   ,  0.   , -0.   , -1.   ],
           [ 0.   ,  0.   ,  0.286,  0.   ],
           [-0.   ,  0.286, -0.   , -0.143],
           [-1.   ,  0.   , -0.143,  0.   ]])
    >>> t_Katz_score(X).round(3)
    array([[ 0.336,  0.007,  0.049, -0.671],
           [ 0.007,  0.021,  0.147, -0.014],
           [ 0.049,  0.147,  0.028, -0.098],
           [-0.671, -0.014, -0.098,  0.343]])
    >>> t_Katz_score(X, beta = 9.75e-1).round(3)
    array([[ 33.541,   1.447,   5.195, -34.401],
           [  1.447,   0.147,   0.526,  -1.484],
           [  5.195,   0.526,   0.889,  -5.328],
           [-34.401,  -1.484,  -5.328,  34.283]])
    >>> t_Katz_score(X, k = 2).round(3)
    array([[ 0.335,  0.029,  0.052, -0.67 ],
           [ 0.029,  0.001,  0.005, -0.015],
           [ 0.052,  0.005,  0.008, -0.104],
           [-0.67 , -0.015, -0.104,  0.343]])
    >>> t_Katz_score(X, beta = 9.75e-1, k = 2).round(3)
    array([[ 33.539,   1.493,   5.207, -34.399],
           [  1.493,   0.065,   0.232,  -1.488],
           [  5.207,   0.232,   0.808,  -5.341],
           [-34.399,  -1.488,  -5.341,  34.283]])

    Katzova ocijena neusmjerenog grafa s težinama.

    >>> gamma, W = t_Katz_score(X, k = 2, compute = False)
    >>> gamma.round(3)
    array([-0.336,  1.022])
    >>> W.round(3)
    array([[-0.699,  0.699],
           [ 0.031,  0.031],
           [-0.108,  0.108],
           [-0.706, -0.706]])

    Dekompozicija Katzove ocijene neusmjerenog grafa s težinama.

    Zabilješke
    ----------
    Za opisane povratne vrijednosti, vrijedi
    `score_k == W_k @ np.diag(gamma_k) @ W_k.T`.

    """

    # Saniraj parametar X.
    if not isinstance(X, _np.ndarray):
        if not (hasattr(X, '__iter__') or hasattr(X, '__array__')):
            raise TypeError('X mora biti klase numpy.ndarray.')
        try:
            X = _np.array(X)
        except (TypeError, ValueError):
            raise TypeError('X mora biti klase numpy.ndarray.')
    if not issubclass(
        X.dtype.type,
        (_numbers.Complex, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        raise TypeError('X mora biti tenzor numerickih vrijednosti.')
    if not X.size:
        raise ValueError('X mora biti neprazni tenzor.')
    if (_np.isnan(X) | _np.isinf(X)).any():
        raise ValueError(
            'X mora sadrzavati samo definirane i konacne vrijednosti.'
        )
    if X.ndim != 2:
        raise ValueError('X mora biti matrica (dvodimenzionalni tenzor).')
    if X.shape[0] != X.shape[1]:
        raise ValueError('X mora biti kvadratna matrica.')
    if not _np.array_equal(X, X.T):
        raise ValueError('X mora biti simetricna matrica.')
    if isinstance(X, _np.matrix):
        X = X.A
    if issubclass(
        X.dtype.type,
        (_numbers.Integral, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        X = X.astype(float)

    # Saniraj parametar beta.
    if not isinstance(beta, _numbers.Real):
        raise TypeError('beta mora biti realni broj.')
    try:
        beta = _copy.deepcopy(float(beta))
    except (TypeError, ValueError):
        raise TypeError('beta mora biti klase float.')
    if _math.isnan(beta) or _math.isinf(beta):
        raise ValueError('beta ne smije biti NaN ili beskonacno.')
    if not (beta > 0.0 and beta < 1.0):
        raise ValueError('beta mora biti u intervalu (0, 1).')

    # Saniraj parametar k.
    if k is not None:
        if not isinstance(k, _numbers.Integral):
            raise TypeError('k mora biti None ili klase int.')
        try:
            k = _copy.deepcopy(int(k))
        except (TypeError, ValueError):
            raise TypeError('k mora biti None ili klase int.')
        if k <= 0:
            raise ValueError('k mora biti strogo pozitivan.')
        if k > min(X.shape):
            raise ValueError('k ne smije nadmasiti dimenzije matrice X.')

    # Saniraj parametar compute.
    if not isinstance(
        compute,
        (_numbers.Integral, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        raise TypeError('compute mora biti klase bool.')
    if compute not in {0, False, 1, True}:
        raise ValueError('compute mora biti laz/istina.')
    try:
        compute = _copy.deepcopy(bool(compute))
    except (TypeError, ValueError):
        raise TypeError('compute mora biti klase bool.')

    # Izračunaj dekompoziciju Katzove ocijene.
    l, W = _np.linalg.eig(X)
    I = _np.flip(_np.argsort(_np.abs(l))).copy(order = 'F')
    l = _np.array(l[I], copy = True, order = 'F')
    W = _np.array(W[:, I], copy = True, order = 'F')
    del I
    if k is None:
        k = (
            1 if _np.isclose(1.0, 1.0 + l[0])
                else int((~_np.isclose(1.0, 1.0 + l / l[0])).sum())
        )
    l = l[:k].copy(order = 'F')
    W = W[:, :k].copy(order = 'F')

    # Vrati Katzovu ocijenu ili njezinu dekompoziciju ovisno o vrijednosti
    # parametra compute.
    return _np.matmul(
        ((1.0 - beta * l) ** -1 - 1.0).copy(order = 'F') * W,
        W.T.copy(order = 'F')
    ).copy(order = 'F') if compute else ((1.0 - beta * l) ** -1 - 1.0, W)

def bt_Katz_score (X, beta = 0.5, k = None, compute = True):
    """
    Izračunaj *truncated bipartite Katz score* matrice bliskosti.

    *Bipartite truncated Katz score* se računa po formuli iz
    [[1]](#bib-dunlavy-10).

    Parametri
    ---------
    X : (M, N) array
        Matrica bliskosti težinskog bipartitnog grafa.  Sve veze moraju imati
        definiranu i konačnu težinu (težina je 0 ako veze nema).  Graf s vezama
        bez težina realizira se vezama težine 1.  Budući da su po pretpostavci
        kontekst redaka i stupaca različiti (na primjer, retci predstavljaju
        osobe, a stupci konferencije), simetričnost matrice `X` ne znači
        simetričnost grafa.  Za Katzovu ocijenu neusmjerenog grafa koristi
        se `t_Katz_score`.

    beta : float in range (0, 1), optional
        Koeficijent relevantnosti duljih puteva (zadana vrijednost je 0.5).

    k : None or int in range [1, min(M, N)], optional
        Broj singularnih vrijednosti matrice X na za računanje ocijene (zadana
        vrijednost je `None`).  Ako je `None`, uzima se

        1.  ako su sve singularne vrijednosti blizu 0, uzima se `k = 1`,
        2.  inače se uzima `k` = # singularnih vrijednosti čiji kvocijent s
            najvećom nije blizu 0.

        Vrijednost `a` je blizu 0 ako vrijedi `np.isclose(1, 1 + a)`.

    compute : boolean, optional
        Ako je laž, povratna vrijednost je dekompozicija Katzove ocijene matrice
        bliskosti `X`; inače je povratna vrijednost Katzova ocijena matrice
        bliskosti `X` (zadana vrijednost je istina).

    Povratne vrijednosti
    --------------------
    Score_k : (M, N) array
        Katzova ocijena matrice bliskosti `X`.  Ova se povratna vrijednost vraća
        ako je `compute` istina.

    U_k : (M, k) array
        Ortogonalna matrica.  Ova se povratna vrijednost vraća ako je `compute`
        laž.

    psi_m_k : (k,) array
        Za konačni niz `s` singularnih vrijednosti matrice `X` poredan silazno,
        `psi_m_k[i]` iznosi
        `beta * s[i] / (1.0 - beta ** 2 * s[i] ** 2)`.  Ova se povratna
        vrijednost vraća ako je `compute` laž.

    psi_p_k : (k,) array
        Za konačni niz `s` singularnih vrijednosti matrice `X` poredan silazno,
        `psi_p_k[i]` iznosi
        `(1.0 - beta ** 2 * s[i] ** 2) ** -1 - 1.0`.  Ova se povratna
        vrijednost vraća ako je `compute` laž.

    V_k : (N, k) array
        Ortogonalna matrica.  Ova se povratna vrijednost vraća ako je `compute`
        laž.

    Iznimke
    -------
    TypeError
        Parametar `X` nije tenzor numeričkih vrijednosti, parametar `beta` nije
        realni broj, parametar `k` nije cijeli broj, parametar `compute` nije
        istinitosna vrijednost.

    ValueError
        Parametar `X` je prazni tenzor, nije matrica, sadrži nedefinirane ili
        beskonačne vrijednosti, parametar `beta` nije u intervalu (0, 1),
        parametar `k` nije u intervalu [1, M], parametar `compute` nije
        laž/istina.

    other
        Iznimke izbačene pozivom `tl.partial_svd(...)` ne hvataju se.

    Vidi također
    ------------
    t_Katz_score

    Primjeri
    --------
    >>> X = [[0, 0, 1, 1],
    ...      [1, 0, 0, 1],
    ...      [0, 1, 1, 1]]
    >>> X = np.array(X, dtype = bool)
    >>> X.astype(int)
    array([[0, 0, 1, 1],
           [1, 0, 0, 1],
           [0, 1, 1, 1]])
    >>> bt_Katz_score(X).round(3)
    array([[-0.545, -0.909, -1.091, -1.636],
           [ 0.364, -0.727, -1.273, -0.909],
           [-0.727, -0.545, -1.455, -2.182]])
    >>> bt_Katz_score(X, beta = 9.75e-1).round(3)
    array([[ 0.057, -0.98 , -0.054,  0.003],
           [-2.356,  1.151,  1.208, -1.148],
           [ 1.151, -0.111, -1.091,  0.06 ]])
    >>> bt_Katz_score(X, k = 2).round(3)
    array([[-0.476, -0.653, -1.201, -1.677],
           [ 0.351, -0.776, -1.252, -0.901],
           [-0.776, -0.724, -1.378, -2.153]])
    >>> bt_Katz_score(X, beta = 9.75e-1, k = 2).round(3)
    array([[ 0.255, -0.246, -0.369, -0.114],
           [-2.393,  1.012,  1.267, -1.126],
           [ 1.012, -0.625, -0.87 ,  0.142]])

    Katzova ocijena bipartitnog grafa bez težina.

    >>> X = X * [-2.0 ** -i for i in range(1, 5)]
    >>> X /= np.abs(X).max()
    >>> X
    array([[-0.   ,  0.   , -0.25 ,  0.125],
           [-1.   ,  0.   , -0.   ,  0.125],
           [-0.   ,  0.5  , -0.25 ,  0.125]])
    >>> bt_Katz_score(X).round(3)
    array([[-0.003,  0.005, -0.13 ,  0.065],
           [-0.67 ,  0.001, -0.001,  0.084],
           [-0.003,  0.272, -0.139,  0.07 ]])
    >>> bt_Katz_score(X, beta = 9.75e-1).round(3)
    array([[ -0.516,   0.063,  -0.3  ,   0.214],
           [-28.756,   0.338,  -0.298,   3.744],
           [ -0.676,   0.723,  -0.393,   0.281]])
    >>> bt_Katz_score(X, k = 2).round(3)
    array([[-0.008,  0.072, -0.047,  0.024],
           [-0.67 ,  0.001, -0.002,  0.085],
           [-0.001,  0.253, -0.163,  0.082]])
    >>> bt_Katz_score(X, beta = 9.75e-1, k = 2).round(3)
    array([[ -0.527,   0.199,  -0.13 ,   0.131],
           [-28.756,   0.337,  -0.3  ,   3.745],
           [ -0.673,   0.684,  -0.442,   0.305]])

    Katzova ocijena bipartitnog grafa s težinama.

    >>> U, psi_m, psi_p, V = bt_Katz_score(X, k = 2, compute = False)
    >>> U.round(3)
    array([[ 0.019, -0.275],
           [ 1.   ,  0.029],
           [ 0.025, -0.961]])
    >>> psi_m.round(3)
    array([0.676, 0.324])
    >>> psi_p.round(3)
    array([0.341, 0.096])
    >>> V.round(3)
    array([[-0.991, -0.049],
           [ 0.012, -0.812],
           [-0.011,  0.522],
           [ 0.129, -0.255]])

    Dekompozicija Katzove ocijene bipartitnog grafa s težinama.

    Zabilješke
    ----------
    Za opisane povratne vrijednosti, vrijedi
    `score_k == U_k @ np.diag(psi_m_k) @ V_k.T`.

    """

    # Saniraj parametar X.
    if not isinstance(X, _np.ndarray):
        if not (hasattr(X, '__iter__') or hasattr(X, '__array__')):
            raise TypeError('X mora biti klase numpy.ndarray.')
        try:
            X = _np.array(X)
        except (TypeError, ValueError):
            raise TypeError('X mora biti klase numpy.ndarray.')
    if not issubclass(
        X.dtype.type,
        (_numbers.Complex, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        raise TypeError('X mora biti tenzor numerickih vrijednosti.')
    if not X.size:
        raise ValueError('X mora biti neprazni tenzor.')
    if (_np.isnan(X) | _np.isinf(X)).any():
        raise ValueError(
            'X mora sadrzavati samo definirane i konacne vrijednosti.'
        )
    if X.ndim != 2:
        raise ValueError('X mora biti matrica (dvodimenzionalni tenzor).')
    if isinstance(X, _np.matrix):
        X = X.A
    if issubclass(
        X.dtype.type,
        (_numbers.Integral, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        X = X.astype(float)

    # Saniraj parametar beta.
    if not isinstance(beta, _numbers.Real):
        raise TypeError('beta mora biti realni broj.')
    try:
        beta = _copy.deepcopy(float(beta))
    except (TypeError, ValueError):
        raise TypeError('beta mora biti klase float.')
    if _math.isnan(beta) or _math.isinf(beta):
        raise ValueError('beta ne smije biti NaN ili beskonacno.')
    if not (beta > 0.0 and beta < 1.0):
        raise ValueError('beta mora biti u intervalu (0, 1].')

    # Saniraj parametar k.
    if k is not None:
        if not isinstance(k, _numbers.Integral):
            raise TypeError('k mora biti None ili klase int.')
        try:
            k = _copy.deepcopy(int(k))
        except (TypeError, ValueError):
            raise TypeError('k mora biti None ili klase int.')
        if k <= 0:
            raise ValueError('k mora biti strogo pozitivan.')
        if k > min(X.shape):
            raise ValueError('k ne smije nadmasiti manju dimenziju matrice X.')

    # Saniraj parametar compute.
    if not isinstance(
        compute,
        (_numbers.Integral, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        raise TypeError('compute mora biti klase bool.')
    if compute not in {0, False, 1, True}:
        raise ValueError('compute mora biti laz/istina.')
    try:
        compute = _copy.deepcopy(bool(compute))
    except (TypeError, ValueError):
        raise TypeError('compute mora biti klase bool.')

    # Izračunaj dekompoziciju Katzove ocijene.
    U = None
    s = None
    V = None
    if k is None:
        U, s, V = _tl.partial_svd(X, int(min(X.shape)))
        k = (
            1 if _np.isclose(1.0, 1.0 + s[0])
                else int((~_np.isclose(1.0, 1.0 + s / s[0])).sum())
        )
        U = _np.array(U[:, :k], copy = True, order = 'F')
        s = _np.array(s[:k], copy = True, order = 'F')
        V = _np.array(V[:k, :].T, copy = True, order = 'F')
    else:
        U, s, V = _tl.partial_svd(X, n_eigenvecs = k)
        U = _np.array(U, copy = True, order = 'F')
        s = _np.array(s, copy = True, order = 'F')
        V = _np.array(V.T, copy = True, order = 'F')

    # Vrati Katzovu ocijenu ili njezinu dekompoziciju ovisno o vrijednosti
    # parametra compute.
    return _np.matmul(
        (beta * s * (1.0 - beta ** 2 * s ** 2) ** -1).copy(order = 'F') * U,
        V.T.copy(order = 'F')
    ).copy(order = 'F') if compute else (
        U,
        beta * s * (1.0 - beta ** 2 * s ** 2) ** -1,
        (1.0 - beta ** 2 * s ** 2) ** -1 - 1.0,
        V
    )

def cp_score (Z, k = None, T0 = None, predict = None, compute = True):
    """
    Izračunaj *CP score* tenzora.

    *CP score* se računa po formuli iz [[1]](#bib-dunlavy-10).

    Parametri
    ---------
    Z : array
        Tenzor numeričkih vrijednosti čiji se *CP score* računa.

    k : None or int in range [1, min(Z.shape)], optional
        Rang komponenti u dekompoziciji tenzora `Z` (zadana vrijednost je
        `None`).  Ako je None, uzima se `k = min(Z.shape)`.

    T0 : None or int in range [1, +inf), optional
        Broj zadnjih stanja koji se uzima u obzir pri predviđanju sljedećih
        stanja (zadana vrijednost je `None`).  Ako je `None`, uzima se
        `T0 = Z.shape[-1]`.

    predict : None or callable, optional
        Funkcija za predviđanje sljedećih stanja (zadana vrijednost je `None`).
        Ako nije `None`, funkcija prima komponentu kao objekt klase
        `numpy.ndarray` oblika `(T0, k)`, a povratna vrijednost mora biti
        također `numpy.array` oblika `(k,)` ili `(n, k)` za neki `n > 0`
        (predviđa se `n` sljedećih stanja).  Ako je `None`, uzima se aritmetička
        sredina po prvoj dimenziji.

    compute : boolean, optional
        Ako je laž, povratna vrijednost je dekompozicija *CP* ocijene matrice
        tenzora `Z`; inače je povratna vrijednost *CP* ocijena tenzora `Z`
        (zadana vrijednost je istina).

    Povratne vrijednosti
    --------------------
    Z_score : array
        *CP* ocijena zadanog tenzora `Z`.  Povratni tenzor dimenzionalnosti je
        za 1 manje od tenzora `Z` ako `predict` predviđa točno jedno stanje, a
        inače je dimenzionalnosti iste kao `Z`.  Prvih `Z.ndim - 1` dimenzija
        jednakih je kao tenzora `Z`, a, ako `predict` predviđa `n > 1` stanja,
        posljednja dimenzija je veličine `n`.  Ako je predikcija array oblika
        () ili (1,), povratna vrijednost je skalar.  Ova se povratna vrijednost
        vraća ako je `compute` istina.

    l : (Z.ndim,) array
        Dekomponiramo li tenzor `Z` na tenzore `cp_components` ranga `k` tako da
        je `j`-ti stupac svake komponente normiran, onda je
        `Z = sum(l[i] * prod(cp_components[:, i]))`.  Ova se povratna vrijednost
        vraća ako je `compute` laž.

    cpd : tuple of arrays
        `cpd` je `tuple` duljine `Z.ndim - 1`, a `cpd[i]` je, kao
        `numpy.ndarray` oblika `(Z.shape[i], k)`, `i`-ta komponenta u *CP*
        dekompoziciji tenzora `Z` na komponente ranga `k`; dodatno, norma svakog
        stupca matrice `cpd[i]` iznosi `1`.  Ova se povratna vrijednost vraća
        ako je `compute` laž.

    p : array
        Rezultat predikcije funkcijom `predict`.  Ako je predikcija rezultirala
        jednodimenzionalnim nizom, `p` će ipak biti dvodimenzionalan (druga
        dimenzija je veličine 1).  Ova se povratna vrijednost vraća ako je
        `compute` laž.

    Iznimke
    -------
    TypeError
        Parametar `Z` nije tenzor numeričkih vrijednosti, parametar `k` nije
        cijeli broj, parametar `T0` nije cijeli broj, parametar `predict` nema
        metodu `__call__`, parametar `compute` nije istinitosna vrijednost,
        predikcija nije tenzor numeričkih vrijednosti.

    ValueError
        Parametar `Z` je prazni tenzor ili sadrži nedefinirane ili beskonačne
        vrijednosti, parametar `k` nije u intervalu [1, `min(Z.shape)`],
        parametar `T0` nije strogo pozitivan, parametar `compute` nije
        laž/istina, predikcija nije niz/matrica odgovarajućih dimenzija.

    other
        Iznimke izbačene pozivom `tl.decomposition.parafac(...)` ne hvataju se.

    Primjeri
    --------
    >>> Z = [[[ 1,  2],
    ...       [ 3,  4],
    ...       [ 5,  6],
    ...       [ 7,  8]],
    ...      [[ 9, 10],
    ...       [11, 12],
    ...       [13, 14],
    ...       [15, 16]],
    ...      [[17, 18],
    ...       [19, 20],
    ...       [21, 22],
    ...       [23, 24]]]
    >>> cp_score(Z).round(3)
    array([[ 1.503,  3.5  ,  5.497,  7.494],
           [ 9.502, 11.5  , 13.499, 15.498],
           [17.5  , 19.5  , 21.501, 23.502]])
    >>> cp_score(Z, k = 1).round(3)
    array([[ 3.836,  4.433,  5.031,  5.628],
           [10.188, 11.775, 13.362, 14.949],
           [16.54 , 19.116, 21.693, 24.269]])
    >>> cp_score(Z, T0 = 1).round(3)
    array([[ 2.061,  4.009,  5.957,  7.905],
           [ 9.982, 11.99 , 13.998, 16.005],
           [17.904, 19.971, 22.038, 24.105]])
    >>> cp_score(Z, k = 1, T0 = 1).round(3)
    array([[ 3.955,  4.571,  5.187,  5.803],
           [10.504, 12.14 , 13.776, 15.412],
           [17.053, 19.709, 22.365, 25.021]])

    *CP* ocijena tenzora.

    >>> np.random.seed(1)
    >>> cp_score(Z, predict = lambda X : np.random.randn(X.shape[-1])).round(3)
    array([[-21.019, -22.13 , -23.241, -24.352],
           [-29.51 , -32.583, -35.657, -38.73 ],
           [-38.001, -43.037, -48.073, -53.109]])
    >>> np.random.seed(1)
    >>> cp_score(Z, predict = lambda X : np.random.randn(*X.shape)).round(3)
    array([[[-21.019,  -3.169],
            [-22.13 ,  -0.827],
            [-23.241,   1.515],
            [-24.352,   3.858]],
           [[-29.51 ,   5.224],
            [-32.583,   7.089],
            [-35.657,   8.954],
            [-38.73 ,  10.819]],
           [[-38.001,  13.617],
            [-43.037,  15.005],
            [-48.073,  16.393],
            [-53.109,  17.78 ]]])

    Ovaj primjer nema stvarnog smisla, osim što demonstrira način poziva s
    promjenom parametra `predict` i mijenjanja količine stanja koja se
    predviđaju.  Prije svakog poziva postavlja se *sjeme* generatora
    pseudo-slučajnih brojeva u biblioteci *NumPy*-ja zato da primjeri budu
    reproducibilni.

    >>> def sqmean (X):
    ...     return np.sqrt(np.mean(np.square(X), axis = 0))
    ...
    >>> cp_score(Z, predict = sqmean).round(3)
    array([[ -1.494,  -3.493,  -5.493,  -7.493],
           [ -9.5  , -11.501, -13.501, -15.502],
           [-17.507, -19.508, -21.51 , -23.511]])

    Primjer kako koristiti vlastitu, složeniju funkciju predikcije (iako se ovaj
    primjer mogao realizirati također koristeći `lambda` funkciju).

    """

    # Saniraj parametar Z.
    if not isinstance(Z, _np.ndarray):
        if not (hasattr(Z, '__iter__') or hasattr(Z, '__array__')):
            raise TypeError('Z mora biti klase numpy.ndarray.')
        try:
            Z = _np.array(Z)
        except (TypeError, ValueError):
            raise TypeError('Z mora biti klase numpy.ndarray.')
    if not issubclass(
        Z.dtype.type,
        (_numbers.Complex, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        raise TypeError('Z mora biti tenzor numerickih vrijednosti.')
    if not Z.ndim:
        raise ValueError('Z mora biti barem jednodimenzionalni tenzor.')
    if not Z.size:
        raise ValueError('Z mora biti neprazni tenzor.')
    if (_np.isnan(Z) | _np.isinf(Z)).any():
        raise ValueError(
            'Z mora sadrzavati samo definirane i konacne vrijednosti.'
        )
    if issubclass(
        Z.dtype.type,
        (_numbers.Integral, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        Z = Z.astype(float)

    # Saniraj parametar k.
    if k is not None:
        if not isinstance(k, _numbers.Integral):
            raise TypeError('k mora biti None ili klase int.')
        try:
            k = _copy.deepcopy(int(k))
        except (TypeError, ValueError):
            raise TypeError('k mora biti None ili klase int.')
        if k <= 0:
            raise ValueError('k mora biti strogo pozitivan.')
        if k > min(Z.shape):
            raise ValueError(
                'k ne smije nadmasiti najmanju dimenziju tenzora Z.'
            )

    # Saniraj parametar T0.
    if T0 is not None:
        if not isinstance(T0, _numbers.Integral):
            raise TypeError('T0 mora biti None ili klase int.')
        try:
            T0 = _copy.deepcopy(int(T0))
        except (TypeError, ValueError):
            raise TypeError('T0 mora biti None ili klase int.')
        if T0 <= 0:
            raise ValueError('T0 mora biti strogo pozitivan.')
        T0 = min(T0, int(Z.shape[-1]))

    # Saniraj parametar predict.
    if predict is None:
        predict = _functools.partial(_np.mean, axis = 0, keepdims = False)
    if not hasattr(predict, '__call__'):
        raise TypeError(
            'Nacin predikcije mora biti zadan funkcijskim objektom.'
        )

    # Saniraj parametar compute.
    if not isinstance(
        compute,
        (_numbers.Integral, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        raise TypeError('compute mora biti klase bool.')
    if compute not in {0, False, 1, True}:
        raise ValueError('compute mora biti laz/istina.')
    try:
        compute = _copy.deepcopy(bool(compute))
    except (TypeError, ValueError):
        raise TypeError('compute mora biti klase bool.')

    # Izračunaj CP dekompoziciju tenzora Z.
    cpd = _copy.deepcopy(
        list(_parafac(Z, int(min(Z.shape)) if k is None else k))
    )
    l = _np.ones(min(Z.shape) if k is None else k, dtype = float, order = 'F')
    for i in iter(range(int(Z.ndim))):
        cpd[i] = _np.array(cpd[i], copy = True, order = 'F')
        if cpd[i].ndim <= 1:
            cpd[i] = cpd[i].reshape((cpd[i].size, 1)).copy(order = 'F')
        for k in iter(range(int(l.size))):
            aux_N = _np.linalg.norm(cpd[i][:, k])
            l[k] *= aux_N
            cpd[i][:, k] /= aux_N
            del aux_N

    # Predvidi vrijednosti posljednje komponente.
    if T0 is not None:
        cpd[-1] = cpd[-1][int(Z.shape[-1] - T0):].copy(order = 'F')
    cpd[-1] = predict(cpd[-1])

    # Saniraj predikciju posljednje komponente.
    if not isinstance(cpd[-1], _np.ndarray):
        if not (hasattr(cpd[-1], '__iter__') or hasattr(cpd[-1], '__array__')):
            raise TypeError('Predikcija mora biti klase numpy.ndarray.')
        try:
            cpd[-1] = _np.array(cpd[-1])
        except (TypeError, ValueError):
            raise TypeError('Predikcija mora biti klase numpy.ndarray.')
    if not issubclass(
        cpd[-1].dtype.type,
        (_numbers.Complex, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        raise TypeError('Predikcija mora biti tenzor numerickih vrijednosti.')
    if cpd[-1].ndim not in {1, 2}:
        raise ValueError(
            'Predikcija mora biti jednodimenzionalna ili dvodimenzionalna.'
        )
    if not cpd[-1].size:
        raise ValueError('Predikcija ne smije biti prazna.')
    if isinstance(cpd[-1], _np.matrix):
        cpd[-1] = cpd[-1].A
    if cpd[-1].ndim == 2:
        cpd[-1] = cpd[-1].T.copy(order = 'C')
    else:
        cpd[-1]= cpd[-1].reshape((cpd[-1].size, 1)).copy(order = 'C')
    if cpd[-1].shape[0] != l.size:
        raise ValueError('Predikcija mora biti koliko i komponenti.')

    # Konvertiraj objekt cpd u tuple.
    cpd = tuple(cpd)

    # Ako se ne traži rekompozicija predikcije, vrati njezinu dekompoziciju.
    if not compute:
        return (l, _copy.deepcopy(cpd[:-1]), cpd[-1].T.copy(order = 'F'))

    # Izračunaj rekompoziciju predikcije.
    # TODO: Ubrzati ovaj dio koda NumPy-jevim "broadcastingom".
    aux_dim = _np.ones(max(Z.ndim - 1, 0), dtype = int, order = 'F')
    S = _np.zeros(
        tuple(_np.concatenate((Z.shape[:-1], [cpd[-1].shape[1]])).tolist()),
        dtype = float,
        order = 'F'
    )
    for k in iter(range(int(l.size))):
        aux_S = l[k] * cpd[-1][k].ravel()
        aux_S = aux_S.reshape(
            tuple(_np.concatenate((aux_dim, [aux_S.size])).tolist())
        )
        for i in iter(range(int(Z.ndim - 1))):
            aux_S = (
                aux_S * cpd[i][:, k].reshape(
                    tuple(
                        _np.concatenate(([Z.shape[i]], aux_dim[i:])).tolist()
                    )
                )
            )
        S = S + aux_S
        del aux_S
    del aux_dim

    # Po potrebi redimenzioniraj predikciju ili ju pretvori u skalar.
    if isinstance(S, _np.ndarray):
        if S.shape == tuple():
            S = S.dtype.type(S)
        elif S.shape[-1] == 1:
            S = S.reshape(S.shape[:-1]).copy(order = 'F')
            if S.shape == tuple():
                S = S.dtype.type(S)
        else:
            S = S.copy(order = 'F')

    # Vrati izračunatu predikciju.
    return S
