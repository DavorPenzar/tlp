# -*- coding: utf-8 -*-

"""
Funkcije za predviđanje veza u mreži kroz vrijeme.

##  Zavisnosti

Kod je izvršiv samo u ***Python* 3** okruženju zbog svojih zavisnosti.

1.  standardna *Python* biblioteka:
    1.  [copy](http://docs.python.org/3/library/copy.html),
    2.  [functools](http://docs.python.org/3/library/functools.html),
    3.  [math](http://docs.python.org/3/library/math.html),
    4.  [numbers](http://docs.python.org/3/library/numbers.html),
2.  ostali paketi:
    1.  [NumPy](http://numpy.org/) &ndash; tenzori su reprezentirani kao objekti
        klase `numpy.ndarray`.
    2.  [TensorLy](http://tensorly.org/) &ndash; dekompozicija tenzora
        realizirana je metodama iz biblioteke paketa *TensorLy* (taj je paket
        podržan samo za *Python* 3 okruženje).

##  Napomene

U ostatku dokumentacije pretpostavlja se da su izvršene linije

```Python
import numpy as np
import tensorLy as tl

from tlp import *
```

##  Literatura

1.  <a class=\"anchor\" id=\"bib-dunlavy-10\"></a> D. M. Dunlavy, T. G. Kolda,
    E. Acar, *Temporal Link Prediction using Matrix and Tensor Factorizations*,
    2010, arXiv: [1005.4006 [math.NA]](http://arxiv.org/abs/1005.4006).
2.  <a class=\"anchor\" id=\"bib-trubetskoy-16\"></a> G. Trubetskoy,
    Holt-Winters Forecasting for Dummies - Part III, dostupno na
    [http://grisha.org/blog/2016/02/17/triple-exponential-smoothing-forecasting-part-iii/](http://grisha.org/blog/2016/02/17/triple-exponential-smoothing-forecasting-part-iii/),
    (lipanj 2019.).

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

class ExponentialSmooth (object):
    """
    Prognoza numeričkih podataka trostrukim eksponencijalnim izglađivanjem.

    Prognoza se računa formulama opisanim u [[2](#bib-trubetskoy-16)].

    """

    def __new__ (cls):
        # Saniraj parametar cls.
        if not issubclass(cls, ExponentialSmooth):
            raise TypeError('cls mora biti ExponentialSmooth.')

        # Kreiraj novi objekt klase ExponentialSmooth.
        instance = super(ExponentialSmooth, cls).__new__(cls)

        # Vrati novi objekt.
        return instance

    def __init__ (self):
        # Saniraj parametar self.
        if not isinstance(self, ExponentialSmooth):
            raise TypeError('self mora biti klase ExponentialSmooth.')

        # Inicijaliziraj self.
        super(ExponentialSmooth, self).__init__()

        # Inicijaliziraj atribute objekta self.
        self._a = _np.zeros(tuple([0]), dtype = float, order = 'F')
        self._n = 0
        self._b = 0.0
        self._s = _np.zeros(tuple([0]), dtype = float, order = 'F')

    def __copy__ (self):
        # Saniraj parametar self.
        if not isinstance(self, ExponentialSmooth):
            raise TypeError('self mora biti klase ExponentialSmooth.')

        # Kreiraj novi objekt klase ExponentialSmooth.
        instance = ExponentialSmooth()

        # Postavi atribute novog objekta na atribute objekta self.
        instance._a = self._a
        instance._n = self._n
        instance._b = self._b
        instance._s = self._s

        # Vrati novi objekt.
        return instance

    def __deepcopy__ (self, memo = dict()):
        # Saniraj parametar self.
        if not isinstance(self, ExponentialSmooth):
            raise TypeError('self mora biti klase ExponentialSmooth.')

        # Kreiraj novi objekt klase ExponentialSmooth.
        instance = ExponentialSmooth()

        # Kopiraj atribute objekta self u atribute novog objekta.
        instance._a = _copy.deepcopy(self._a, memo)
        instance._n = _copy.deepcopy(self._n, memo)
        instance._b = _copy.deepcopy(self._b, memo)
        instance._s = _copy.deepcopy(self._s, memo)

        # Vrati novi objekt.
        return instance

    def __repr__ (self):
        # Saniraj parametar self.
        if not isinstance(self, ExponentialSmooth):
            raise TypeError('self mora biti klase ExponentialSmooth.')

        # Vrati tekstualnu reprezentaciju objekta self.
        return '<{class_name:s}: ({series_shape:s}; {season_length:d})>'.format(
            class_name = self.__class__.__name__,
            series_shape = repr(tuple(self._a.shape)),
            season_length = self._n
        )

    def fit (self, a, n):
        """
        Pripremi prognoziranje vrijednosti tenzora `a` s periodom duljine `n`.

        Parametri
        ---------
        a : array
            Tenzor čije prognoze će se računati (po zadnjoj dimenziji).

        n : int in range [1, a.shape[-1] // 2]
            Duljina perioda u tenzoru `a`.

        Povratne vrijednosti
        --------------------
        self : ExponentialSmooth
            Povratna vrijednost je `self`.

        Iznimke
        -------
        TypeError
            Parametar `a` nije nije tenzor numeričkih vrijednosti, parametar `n`
            nije vijeli broj.

        ValueError
            Parametar `a` je prazni tenzor, sadrži nedefinrane ili beskonačne
            vrijednosti, parametar `n` nije u intervalu [1, `a.shape[-1] // 2`].

        """

        # Saniraj parametar self.
        if not isinstance(self, ExponentialSmooth):
            raise TypeError('self mora biti klase ExponentialSmooth.')

        # Saniraj parametar a.
        if not isinstance(a, _np.ndarray):
            if not (hasattr(a, '__iter__') or hasattr(a, '__array__')):
                raise TypeError('a mora biti klase numpy.ndarray.')
            try:
                a = _np.array(a)
            except (TypeError, ValueError, AttributeError):
                raise TypeError('a mora biti klase numpy.ndarray.')
        if not issubclass(
            a.dtype.type,
            (_numbers.Complex, int, bool, _np.bool, _np.bool8, _np.bool_)
        ):
            raise TypeError('a mora biti tenzor numerickih vrijednosti.')
        if not a.ndim:
            raise ValueError('a mora biti barem jednodimenzionalni tenzor.')
        if not a.size:
            raise ValueError('a mora biti neprazni tenzor.')
        if (_np.isnan(a) | _np.isinf(a)).any():
            raise ValueError(
                'a mora sadrzavati samo definirane i konacne vrijednosti.'
            )
        if isinstance(a, _np.matrix):
            a = a.A

        # Saniraj parametar n.
        if isinstance(n, _np.ndarray):
            if n.shape == tuple() or n.size == 1:
                n = n.ravel()
                n = n.dtype.type(n[0])
        if not isinstance(n, _numbers.Integral):
            raise TypeError('n mora biti klase int.')
        try:
            n = _copy.deepcopy(int(n))
        except (TypeError, ValueError, AttributeError):
            raise TypeError('n mora biti klase int.')
        if n <= 0:
            raise ValueError('n mora biti strogo pozitivan.')
        if n > a.shape[-1] // 2:
            raise ValueError('n ne smije biti veci od a.shape[-1] // 2.')

        # Spremi kopije parametara u atribute objekta self.  Jednostavnosti
        # računa radi, posljednja dimenzija tenzora a prebacuje se na prvo
        # mjesto.
        self._a = _np.moveaxis(a, -1, 0).copy(order = 'F')
        self._n = _copy.deepcopy(n)

        # Izračunaj pomoćne varijable broj cijelih perioda u tenzoru a i srednje
        # vrijednosti perioda.
        N = int(_math.floor(float(self._a.shape[0]) / self._n))
        avg = _np.array(
            [
                self._a[
                    i * self._n:_np.minimum((i + 1) * self._n, self._a.shape[0])
                ].mean(axis = 0) for i in iter(range(N))
            ],
            order = 'F'
        )

        # Izračunaj (i spremi u atribute objekta self) inicijalni trend i
        # inicijalne sezonalne komponente tenzora a.
        self._b = (
            (self._a[self._n:2 * self._n] - self._a[:self._n]).sum(axis = 0) /
            float(self._n) ** 2
        )
        if isinstance(self._b, _np.ndarray):
            if self._b.shape == tuple() or self._b.size == 1:
                self._b = self._b.ravel()
                self._b = self._b.dtype.type(self._b[0])
        self._s = _np.array(
            [
                (self._a[i:N * self._n:self._n] - avg).sum(axis = 0)
                    for i in iter(range(self._n))
            ],
            order = 'F'
        ) / float(N)

        # Vrati self.
        return self

    def predict (self, k = 1, theta = 0.5):
        """
        Predvidi vrijednosti tenzora.

        Parametri
        ---------
        k : int in range [1, +inf), optional
            Broj vrijednosti koje se predviđa (zadana vrijednost je 1).

        theta : float in range [0, 1] or array of such or tuple of 3 of such
            Parametri eksponencijalnog izglađivanja (zadana vrijednost je 0.5).
            Ako je jedinstvena vrijednost, uzima se
            `theta = (theta, theta, theta)`.  Vrijednost `theta[0]` koeficijent
            je izglađivanja visine, vrijednost `theta[1]` koeficijent je
            izglađivanja trenda, a vrijednost `theta[2]` je koeficijent
            izglađivanja sezonalnih komponenti.  Ako je neka od vrijednosti
            `theta` klase `numpy.ndarray`, pri računu se provode standardna
            pravila *broadcastinga*.

        Povratne vrijednosti
        --------------------
        y : array
            Predviđenih `k` vrijednosti tenzora `a`.  Prvih `a.ndim - 1`
            dimenzija jednakih je veličina kao tenzora `a`, a posljednja
            dimenzija je veličine `k`.  Ako je `k == 1`, onda je povratna
            vrijednost dimenzionalnosti `a.ndim - 1`.  Ako je `a`
            jednodimenzionalni tenzor i `k == 1`, onda je povratna vrijednost
            skalar.

        Iznimke
        -------
        TypeError
            Parametar `k` nije cijeli broj, parametar `theta` sadrži vrijednost
            koja nije realni broj ni tenzor realnih brojeva.

        ValueError
            Parametar `k` nije u intervalu [1, +inf), parametar `theta` sadrži
            vrijednost koja nije u intervalu [0, 1].

        other
            Ako je neka od vrijednosti `theta` klase `numpy.ndarray` takav da
            se standardna pravila *broadcastinga* krše, iznimke koje takav
            račun izbacuje ne hvataju se.

        """

        # Saniraj parametar self.
        if not isinstance(self, ExponentialSmooth):
            raise TypeError('self mora biti klase ExponentialSmooth.')

        # Saniraj parametar k.
        if isinstance(k, _np.ndarray):
            if k.shape == tuple() or k.size == 1:
                k = k.ravel()
                k = k.dtype.type(k[0])
        if not isinstance(k, _numbers.Integral):
            raise TypeError('k mora biti klase int.')
        try:
            k = _copy.deepcopy(int(k))
        except (TypeError, ValueError, AttributeError):
            raise TypeError('k mora biti klase int.')
        if k <= 0:
            raise ValueError('k mora biti strogo pozitivan.')

        # Saniraj parametar theta.
        if not isinstance(theta, tuple):
            if isinstance(theta, _np.ndarray):
                theta = _copy.deepcopy((theta, theta, theta))
            else:
                if hasattr(theta, '__iter__'):
                    theta = _copy.deepcopy(tuple(theta))
                else:
                    theta = _copy.deepcopy((theta, theta, theta))
        else:
            theta = _copy.deepcopy(theta)
        if len(theta) != 3:
            theta = _copy.deepcopy((theta, theta, theta))
        theta = list(theta)
        for i in iter(range(3)):
            if (
                not isinstance(theta[i], _np.ndarray) and
                (
                    hasattr(theta[i], '__iter__') or
                    hasattr(theta[i], '__array__')
                )
            ):
                    try:
                        theta[i] = _np.array(theta[i])
                    except (TypeError, ValueError, AttributeError):
                        raise TypeError(
                            'theta mora biti skalar ili klase numpy.ndarray.'
                        )
            if isinstance(theta[i], _np.ndarray):
                if theta[i].shape == tuple() or theta[i].size == 1:
                    theta[i] = theta[i].ravel()
                    theta[i] = theta[i].dtype.type(theta[i][0])
            if isinstance(theta[i], _np.ndarray):
                if not issubclass(
                    theta[i].dtype.type,
                    (_numbers.Real, int, bool, _np.bool, _np.bool8, _np.bool_)
                ):
                    raise TypeError(
                        'theta mora biti tenzor realnih vrijednosti.'
                    )
                if issubclass(
                    theta[i].dtype.type,
                    (int, bool, _np.bool, _np.bool8, _np.bool_)
                ):
                    theta[i] = theta[i].astype(float)
                if not theta[i].ndim:
                    raise ValueError(
                        'thta mora biti barem jednodimenzionalni tenzor.'
                    )
                if not theta[i].size:
                    raise ValueError('theta mora biti neprazni tenzor.')
                if (_np.isnan(theta[i]) | _np.isinf(theta[i])).any():
                    raise ValueError(
                        'theta mora sadrzavati samo definirane i konacne '
                        'vrijednosti.'
                    )
                if ((theta[i] < 0.0) | (theta[i] > 1.0)).any():
                    raise ValueError(
                        'theta mora sadrzavati vrijednosti u intervalu [0, 1].'
                    )
                if isinstance(theta[i], _np.matrix):
                    theta[i] = theta[i].A
            else:
                if not (isinstance(theta[i], _numbers.Real)):
                    raise TypeError('theta mora biti klase float.')
                try:
                    theta[i] = _copy.deepcopy(float(theta[i]))
                except (TypeError, ValueError, AttributeError):
                    raise TypeError('theta mora biti klase float.')
                if _math.isnan(theta[i]) or _math.isinf(theta[i]):
                    raise ValueError('theta ne smije biti NaN ili beskonacno.')
                if theta[i] < 0.0 or theta[i] > 1.0:
                    raise ValueError('theta mora biti u intervalu [0, 1].')
        theta = tuple(theta)

        # Izračunaj 1 - theta po komponentama.
        one_min_theta = (1.0 - theta[0], 1.0 - theta[1], 1.0 - theta[2])

        # Eksponencijalnim izglađivanjem izračunaj tenzor y duljine
        # a.shape[-1] + k.
        y = None
        if self._a.ndim == 1:
            y = _np.zeros(self._a.size + k, dtype = self._s.dtype, order = 'F')
        else:
            y = _np.zeros(
                tuple(
                    _np.concatenate(
                        ([self._a.shape[0] + k], self._a.shape[1:])
                    ).tolist()
                ),
                dtype = self._s.dtype,
                order = 'F'
        )
        b = _copy.deepcopy(self._b)
        s = _copy.deepcopy(self._s)
        l = (_copy.deepcopy(self._a[0]), _copy.deepcopy(self._a[0]))
        for i in iter(range(1, int(self._a.shape[0]))):
            val = _copy.deepcopy(self._a[i])
            l = (
                l[1],
                (
                    theta[0] * (val - s[i % self._n]) +
                    one_min_theta[0] * (l[1] + b)
                )
            )
            b = theta[1] * (l[1] - l[0]) + one_min_theta[1] * b
            s[i % self._n] = (
                theta[2] * (val - l[1]) + one_min_theta[2] * s[i % self._n]
            )
            y[i] = l[1] + b + s[i % self._n]
        for i in iter(range(int(self._a.shape[0]), int(y.shape[0]))):
            y[i] = (l[1] + (i - self._a.shape[0] + 1) * b) + s[i % self._n]

        # Spremi samo posljednjih k vrijednosti u tenzoru y.
        y = y[int(self._a.shape[0]):].copy(order = 'F')

        # Po potrebi pojednostavi dimenzionalnost tenzora y odnosno pretvori ga
        # u skalar.
        if isinstance(y, _np.ndarray):
            if k == 1:
                if y.ndim == 1:
                    y = y.dtype.type(y[0])
                else:
                    y.shape = y.shape[1:]
            else:
                y = _np.moveaxis(y, 0, -1).copy(order = 'F')

        # Vrati izračunati tenzor y.
        return y

    @property
    def series_ (self):
        """
        Tenzor čije prognoze se računaju eksponencijalnim izglađivanjem.

        """

        # Saniraj parametar self.
        if not isinstance(self, ExponentialSmooth):
            raise TypeError('self mora biti klase ExponentialSmooth.')

        # Vrati tenzor.
        return _np.moveaxis(self._a, 0, -1).copy(order = 'F')

    @property
    def season_length_ (self):
        """
        Duljina perioda vrijednosti tenzora.

        """

        # Saniraj parametar self.
        if not isinstance(self, ExponentialSmooth):
            raise TypeError('self mora biti klase ExponentialSmooth.')

        # Vrati duljinu perioda tenzora.
        return _copy.deepcopy(self._n)

    @property
    def initial_trend_ (self):
        """
        Inicijalni trend tenzora.

        """

        # Saniraj parametar self.
        if not isinstance(self, ExponentialSmooth):
            raise TypeError('self mora biti klase ExponentialSmooth.')

        # Vrati inicijalni trend tenzora.
        return _copy.deepcopy(self._b)

    @property
    def initial_seasonal_components_ (self):
        """
        Inicijalne sezonalne komponente tenzora.

        """

        # saniraj parametar self.
        if not isinstance(self, ExponentialSmooth):
            raise TypeError('self mora biti klase ExponentialSmooth.')

        # Vrati inicijalne sezonalne komponente tenzora.
        return _copy.deepcopy(self._s)

def cwt (Z, theta = 0.5, norm = False):
    """
    Izračunaj *collapsed weighted tensor* (*CWT*).

    *CWT* se računa po formuli iz [[1]](#bib-dunlavy-10).

    Parametri
    ---------
    Z : array
        Tenzor numeričkih vrijednosti čiji se *CWT* računa.

    theta : float in range [0 to 1), optional
        Parametar *gubitka* relevantnosti stanja kroz vrijeme (zadana vrijednost
        je 0.5).

    norm : boolean, optional
        Ako je istina, povratni tenzor je težinska sredina umjesto obične sume,
        to jest, povratna vrijednost je podijeljena sa
        `sum((1 - theta) ** i for i in range(Z.shape[-1]))` (zadana vrijednost
        je laž).

    Povratne vrijednosti
    --------------------
    Z_cwt : array
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
    if isinstance(Z, _np.matrix):
        Z = Z.A
    if issubclass(
        Z.dtype.type,
        (_numbers.Integral, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        Z = Z.astype(float)

    # Saniraj parametar theta.
    if isinstance(theta, _np.ndarray):
        if theta.shape == tuple() or theta.size == 1:
            theta = theta.ravel()
            theta = theta.dtype.type(theta[0])
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
    if isinstance(norm, _np.ndarray):
        if norm.shape == tuple() or norm.size == 1:
            norm = norm.ravel()
            norm = norm.dtype.type(norm[0])
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
    one_min_theta = _np.flip(
        (1.0 - theta) ** _np.arange(Z.shape[-1], dtype = int)
    ).copy(order = 'F')

    # Izračunaj kompresijsku sumu.
    Z_compressed = (one_min_theta * Z).sum(axis = -1).copy(order = 'F')

    # Ako je norm istina, podijeli kompresijsku sumu sa _theta.sum().
    if norm:
        Z_compressed /= one_min_theta.sum()

    # Po potrebi pretvori Z u skalar.
    if isinstance(Z_compressed, _np.ndarray):
        if Z_compressed.shape == tuple():
            Z_compressed = Z_compressed.ravel()
            Z_compressed = Z_compressed.dtype.type(Z_compressed[0])

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

    k : None or int in range [1, min(M, N)], optional
        Broj singularnih vrijednosti matrice X za računanje ocijene (zadana
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

    V_k : (N, k) array
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
    if isinstance(k, _np.ndarray):
        if k.shape == tuple() or k.size == 1:
            k = k.ravel()
            k = k.dtype.type(k[0])
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
    if isinstance(compute, _np.ndarray):
        if compute.shape == tuple() or compute.size == 1:
            compute = compute.ravel()
            compute = compute.dtype.type(compute[0])
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
        Broj singularnih vrijednosti matrice X za računanje ocijene (zadana
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
        `(1 - beta * l[i]) ** -1 - 1`.  Ova se povratna vrijednost vraća ako je
        `compute` laž.

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
        intervalu (0, 1), parametar `k` nije u intervalu [1, `M`], parametar
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
    if isinstance(beta, _np.ndarray):
        if beta.shape == tuple() or beta.size == 1:
            beta = beta.ravel()
            beta = beta.dtype.type(beta[0])
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
    if isinstance(k, _np.ndarray):
        if k.shape == tuple() or k.size == 1:
            k = k.ravel()
            k = k.dtype.type(k[0])
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
    if isinstance(compute, _np.ndarray):
        if compute.shape == tuple() or compute.size == 1:
            compute = compute.ravel()
            compute = compute.dtype.type(compute[0])
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
        Broj singularnih vrijednosti matrice X za računanje ocijene (zadana
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
        `beta * s[i] * (1 - beta ** 2 * s[i] ** 2) ** -1`.  Ova se povratna
        vrijednost vraća ako je `compute` laž.

    psi_p_k : (k,) array
        Za konačni niz `s` singularnih vrijednosti matrice `X` poredan silazno,
        `psi_p_k[i]` iznosi
        `(1 - beta ** 2 * s[i] ** 2) ** -1 - 1`.  Ova se povratna vrijednost
        vraća ako je `compute` laž.

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
        parametar `k` nije u intervalu [1, `M`], parametar `compute` nije
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
    if isinstance(beta, _np.ndarray):
        if beta.shape == tuple() or beta.size == 1:
            beta = beta.ravel()
            beta = beta.dtype.type(beta[0])
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
    if isinstance(k, _np.ndarray):
        if k.shape == tuple() or k.size == 1:
            k = k.ravel()
            k = k.dtype.type(k[0])
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
    if isinstance(compute, _np.ndarray):
        if compute.shape == tuple() or compute.size == 1:
            compute = compute.ravel()
            compute = compute.dtype.type(compute[0])
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
        `None`).  Ako je `None`, uzima se `k = min(Z.shape)`.

    T0 : None or int in range [1, +inf), optional
        Broj zadnjih stanja koji se uzima u obzir pri predviđanju sljedećih
        stanja (zadana vrijednost je `None`).  Ako je `None`, uzima se
        `T0 = Z.shape[-1]`.

    predict : None or callable, optional
        Funkcija za predviđanje sljedećih stanja (zadana vrijednost je `None`).
        Ako nije `None`, funkcija prima komponentu kao objekt klase
        `numpy.ndarray` oblika `(T0, k)` osim ako je `k == 1` (u tom slučaju je
        argument `numpy.ndarray` oblika `(T0,)` za `T0 != 1` odnosno jedinstveni
        skalar za `T0 == 1`), a povratna vrijednost mora biti također
        `numpy.array` oblika `(k,)` ili `(n, k)` za neki `n > 0` (predviđa se
        `n` sljedećih stanja); ako je `k == 1`, povratna vrijednost smije biti
        jedinstveni skalar.  Ako je `None`, uzima se aritmetička sredina po
        prvoj dimenziji.

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
        `()` ili `(1,)`, povratna vrijednost je skalar.  Ova se povratna
        vrijednost vraća ako je `compute` istina.

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
        jednodimenzionalnim nizom ili skalarom, `p` će ipak biti
        dvodimenzionalan (druga dimenzija je veličine 1, za skalarnu predikciju
        će i prva dimenzija biti veličine 1).  Ova se povratna vrijednost vraća
        ako je `compute` laž.

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
    if isinstance(Z, _np.matrix):
        Z = Z.A
    if issubclass(
        Z.dtype.type,
        (_numbers.Integral, int, bool, _np.bool, _np.bool8, _np.bool_)
    ):
        Z = Z.astype(float)

    # Saniraj parametar k.
    if isinstance(k, _np.ndarray):
        if k.shape == tuple() or k.size == 1:
            k = k.ravel()
            k = k.dtype.type(k[0])
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
    if isinstance(T0, _np.ndarray):
        if T0.shape == tuple() or T0.size == 1:
            T0 = T0.ravel()
            T0 = T0.dtype.type(T0[0])
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
    if isinstance(predict, _np.ndarray):
        if predict.shape == tuple() or predict.size == 1:
            predict = predict.ravel()
            predict = predict.dtype.type(predict[0])
    if predict is None:
        if ((T0 is None and Z.shape[-1] == 1) or T0 == 1) and k == 1:
            predict = lambda x : x
        else:
            predict = _functools.partial(_np.mean, axis = 0, keepdims = False)
    if not hasattr(predict, '__call__'):
        raise TypeError(
            'Nacin predikcije mora biti zadan funkcijskim objektom.'
        )

    # Saniraj parametar compute.
    if isinstance(compute, _np.ndarray):
        if compute.shape == tuple() or compute.size == 1:
            compute = compute.ravel()
            compute = compute.dtype.type(compute[0])
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
            cpd[i] = cpd[i].ravel()
            cpd[i] = cpd[i].reshape((cpd[i].size, 1)).copy(order = 'F')
        for k in iter(range(int(l.size))):
            aux_N = _np.linalg.norm(cpd[i][:, k])
            if aux_N:
                l[k] *= aux_N
                cpd[i][:, k] /= aux_N
            else:
                l[k] = 0
                cpd[i][:, k] = 0
            del aux_N
    I = _np.flip(_np.argsort(l)).copy(order = 'F')
    l = l[I].copy(order = 'F')
    for i in iter(range(int(Z.ndim))):
        cpd[i] = cpd[i][:, I].copy(order = 'F')
    del I

    # Predvidi vrijednosti posljednje komponente.
    if T0 is not None:
        cpd[-1] = cpd[-1][int(Z.shape[-1] - T0):].copy(order = 'F')
    if cpd[-1].shape[1] == 1:
        cpd[-1] = cpd[-1].ravel().copy(order = 'F')
    if cpd[-1].size == 1:
        cpd[-1] = cpd[-1].ravel().copy(order = 'F')
        cpd[-1] = cpd[-1].dtype.type(cpd[-1][0])
    cpd[-1] = predict(cpd[-1])

    # Saniraj predikciju posljednje komponente.
    if not isinstance(cpd[-1], _np.ndarray):
        if not (hasattr(cpd[-1], '__iter__') or hasattr(cpd[-1], '__array__')):
            cpd[-1] = _np.array([cpd[-1]])
        try:
            cpd[-1] = _np.array(cpd[-1])
        except (TypeError, ValueError):
            raise TypeError('Predikcija mora biti klase numpy.ndarray.')
    if cpd[-1].shape == tuple() or cpd[-1].size == 1:
        cpd[-1] = cpd[-1].ravel()
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
    if cpd[-1].size == 1:
        cpd[-1] = cpd[-1][0] * _np.ones(l.size, dtype = int, order = 'F')
    if isinstance(cpd[-1], _np.matrix):
        cpd[-1] = cpd[-1].A
    if cpd[-1].ndim == 2:
        cpd[-1] = cpd[-1].T.copy(order = 'C')
    else:
        cpd[-1]= cpd[-1].reshape((cpd[-1].size, 1)).copy(order = 'C')
    if cpd[-1].shape[0] != l.size:
        if l.size == 1 and cpd[-1].ndim == 1 and cpd[-1].size != 1:
            cpd[-1] = cpd[-1] = cpd[-1].reshape((1, cpd[-1].size))
        else:
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
            S = S.ravel()
            S = S.dtype.type(S[0])
        elif S.shape[-1] == 1:
            S = S.reshape(S.shape[:-1]).copy(order = 'F')
            if S.shape == tuple():
                S = S.ravel()
                S = S.dtype.type(S[0])
        else:
            S = S.copy(order = 'F')

    # Vrati izračunatu predikciju.
    return S

def rand_fft (n, mu = 0.0, sigma = _math.sqrt(0.5)):
    """
    Generiraj koeficijente slučajnog jednodimenzionalne diskretne Fourireove transformacije.

    Funkcija normalnom distribucijom generira parametar koji se može
    proslijediti funkcijama `numpy.fft.ifft` i `numpy.fft.irfft` kao
    koeficijenti jednodimenzionalne diskretne Fourireove transformacije.

    Ako su `mu` i `sigma` `tuple`-ovi duljine 2, povratna vrijednost
    ekvivalentna je pozivu (osim što se za `n == 1` ekstrahira jedinstveni
    skalar)

    >>> np.array(sigma[0] * np.random.randn(n) + mu[0] + (sigma[1] * np.random.randn(n) + mu[1]) * complex(0.0, 1.0), dtype = complex)

    Parametri
    ---------
    n : int in range [0, +inf)
        Broj koeficijenata koji se generiraju.

    mu : complex or (n,) array or tuple of 2 of such
        Očekivanje koeficijenata (zadana vrijednost je 0).  Ako je jedinstveni
        broj ili jedinstveni niz duljine `n`, uzima se `mu = (mu, mu)`.

    sigma : complex or (n,) array or tuple of 2 of such
        "Standardna devijacija" koeficijenata (zadana vrijednost je
        sqrt(2) / 2).  Može biti 0, negativno ili čak kompleksni broj s
        imaginarnim dijelom različitim od 0.  Ako je jedinstveni broj ili
        jedinstveni niz duljine `n`, uzima se `sigma = (sigma, sigma)`.

    Povratne vrijednosti
    --------------------
    F : (n,) array
        Niz od `n` kompleksnih brojeva čiji su realni i imaginarni dijelovi
        distribuirani sa zadanim parametrima.  Ako je `n == 1`, povratna
        vrijednost je skalar.

    Iznimke
    -------
    TypeError
        Parametar `n` nije cijeli broj, parametar `mu` sadrži vrijednost koja
        nije kompleksni broj, parametar `sigma` sadrži vrijednost koja nije
        kompleksni broj.

    ValueError
        Parametar `n` je negativan, parametar `mu` sadrži niz koji nije duljine
        `n`, parametar `sigma` sadrži niz koji nije duljine `n`, parametar `mu`
        sadrži nedefinirane ili beskonačne vrijednosti, parametar `sigma` sadrži
        nedefinirane ili beskonačne vrijednosti.

    """

    # Saniraj parametar n.
    if not isinstance(n, _numbers.Integral):
        raise TypeError('n mora biti cijeli broj.')
    try:
        n = _copy.deepcopy(int(n))
    except (TypeError, ValueError, AttributeError):
        raise TypeError('n mora biti klase int.')
    if n < 0:
        raise ValueError('n mora biti nenegativan.')

    # Saniraj parametre mu i sigma.
    if not isinstance(mu, tuple):
        if isinstance(mu, _np.ndarray):
            mu = _copy.deepcopy((mu, mu))
        else:
            if hasattr(mu, '__iter__'):
                mu = _copy.deepcopy(tuple(mu))
            else:
                mu = _copy.deepcopy((mu, mu))
    else:
        mu = _copy.deepcopy(mu)
    if not isinstance(sigma, tuple):
        if isinstance(sigma, _np.ndarray):
            sigma = _copy.deepcopy((sigma, sigma))
        else:
            if hasattr(sigma, '__iter__'):
                sigma = _copy.deepcopy(tuple(sigma))
            else:
                sigma = _copy.deepcopy((sigma, sigma))
    else:
        mu = _copy.deepcopy(mu)
    if len(mu) != 2:
        mu = _copy.deepcopy((mu, mu))
    if len(sigma) != 2:
        sigma = _copy.deepcopy((sigma, sigma))
    mu = list(mu)
    sigma = list(sigma)
    for i in iter(range(2)):
        if isinstance(mu[i], _np.ndarray):
            if mu[i].shape == tuple() or mu[i].size == 1:
                mu[i] = mu[i].ravel()
                mu[i] = mu[i].dtype.type(mu[i][0])
        if (
            not isinstance(mu[i], _np.ndarray) and
            (hasattr(mu[i], '__iter__') or hasattr(mu[i], '__array__'))
        ):
            try:
                mu[i] = _np.array(mu[i])
            except (TypeError, ValueError, AttributeError):
                raise TypeError('mu mora biti skalar ili klase numpy.ndarray.')
        if isinstance(mu[i], _np.ndarray):
            if not issubclass(mu[i].dtype.type, _numbers.Complex):
                raise TypeError('mu mora biti niz numerickih vrijednosti.')
            if mu[i].ndim != 1:
                raise ValueError('mu mora biti jednodimenzionalni niz.')
            if mu[i].size != 1 and mu[i].size != n:
                raise ValueError('mu mora biti duljine n.')
            mu[i] = mu[i].astype(complex)
            if (_np.isnan(mu[i]) | _np.isinf(mu[i])).any():
                raise ValueError(
                    'mu mora sadrzavati samo definirane i konacne vrijednosti.'
                )
        else:
            if not isinstance(mu[i], _numbers.Complex):
                raise TypeError('mu mora biti numericki skalar')
            try:
                mu[i] = _copy.deepcopy(complex(mu[i]))
            except (TypeError, ValueError, AttributeError):
                raise TypeError('')
            if (
                _math.isnan(mu[i].real) or
                _math.isnan(mu[i].imag) or
                _math.isinf(mu[i].real) or
                _math.isinf(mu[i].imag)
            ):
                raise ValueError('mu ne smije biti NaN ili beskonacno.')
        if isinstance(sigma[i], _np.ndarray):
            if sigma[i].shape == tuple() or sigma[i].size == 1:
                sigma[i] = sigma[i].ravel()
                sigma[i] = sigma[i].dtype.type(sigma[i][0])
        if (
            not isinstance(sigma[i], _np.ndarray) and
            (hasattr(sigma[i], '__iter__') or hasattr(sigma[i], '__array__'))
        ):
            try:
                sigma[i] = _np.array(sigma[i])
            except (TypeError, ValueError, AttributeError):
                raise TypeError(
                    'sigma mora biti skalar ili klase numpy.ndarray.'
                )
        if isinstance(sigma[i], _np.ndarray):
            if not issubclass(sigma[i].dtype.type, _numbers.Complex):
                raise TypeError('sigma mora biti niz numerickih vrijednosti.')
            if sigma[i].ndim != 1:
                raise ValueError('sigma mora biti jednodimenzionalni niz.')
            if sigma[i].size != 1 and sigma[i].size != n:
                raise ValueError('sigma mora biti duljine n.')
            sigma[i] = sigma[i].astype(complex)
            if (_np.isnan(sigma[i]) | _np.isinf(sigma[i])).any():
                raise ValueError(
                    'sigma mora sadrzavati samo definirane i konacne '
                    'vrijednosti.'
                )
        else:
            if not isinstance(sigma[i], _numbers.Complex):
                raise TypeError('sigma mora biti numericki skalar')
            try:
                sigma[i] = _copy.deepcopy(complex(sigma[i]))
            except (TypeError, ValueError, AttributeError):
                raise TypeError('')
            if (
                _math.isnan(sigma[i].real) or
                _math.isnan(sigma[i].imag) or
                _math.isinf(sigma[i].real) or
                _math.isinf(sigma[i].imag)
            ):
                raise ValueError('sigma ne smije biti NaN ili beskonacno.')
    mu = tuple(mu)
    sigma = tuple(sigma)

    # Generiraj "realne" i "imaginarne" dijelove koeficijenata.
    Fr = (sigma[0] * _np.random.randn(n)) + mu[0]
    Fi = (sigma[1] * _np.random.randn(n)) + mu[1]

    # Generiraj niz koeficijenata jednodimenzionalne diskretne Fourireove
    # transformacije.
    F = _np.array(Fr + Fi * complex(0.0, 1.0), dtype = complex, order = 'F')

    # Po potrebi pretvori F u skalar.
    if isinstance(F, _np.ndarray):
        if F.shape == tuple() or F.size == 1:
            F = F.ravel()
            F = F.dtype.type(F[0])

    # Vrati generirani niz F.
    return F
