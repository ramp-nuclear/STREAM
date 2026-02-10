"""Tools to generate UQ models

"""
from functools import reduce
from typing import Callable, TypeVar, Iterable

import numpy as np
from dask import delayed
from dask.delayed import Delayed
from pandas import DataFrame, concat

from stream.units import Value, Array1D, Array2D
from .uncertainty import Uncertuple, Uncertainty

Model = Callable[[...], DataFrame]
DelayedModel = Callable[[...], Delayed]
ModelFactory = Callable[[...], tuple[Delayed, Delayed]]
T = TypeVar("T", bound=Value, covariant=True)


def _vector_step(x: T, h: T) -> Iterable[T]:
    """Generate finite-difference step values ``x+h``

    Parameters
    ----------
    x : T (Base ``Value``)
        Vector to perturb
    h : T
        Perturbations required in each entry of x. x and h have to have the same shape.

    Yields
    ------
    from Iterable[T]
        If ``x`` and ``h`` are vectors of length ``n``,
        then each element is added separately, creating a generator with ``n`` elements,
        schematically ``step[i] = x[i] + h_i, step[j!=i]=x for each i``.

    Examples
    --------
    >>> list(_vector_step(5., 0.5))
    [5.5]
    >>> list(_vector_step(np.arange(2), np.ones(2)))
    [array([1., 1.]), array([0., 2.])]
    >>> list(_vector_step(5., 1))
    [6.0]
    """
    match x, h:
        case int() | float() | np.float64(), int() | float() | np.float64():
            yield x + h
        case np.ndarray(ndim=1), np.ndarray(ndim=1) if x.shape == h.shape:
            yield from np.diag(h) + x
        case _:
            raise TypeError("x and h must be of the same type, float or 1D ndarray: "
                            f"{x = }, {h = }, {type(x) = }, {type(h) = }")


def default_deriv_step(x: T) -> T:
    """The default way to make an infinitesimal change in x.

    Parameters
    ----------
    x: T
        The value to change things around

    """
    return 1e-12 + 1e-6 * np.abs(x)


def _as_matrix(nominal,
               perturbed_solutions: Iterable[Array1D],
               perturbations: Array1D,
               length: int = None) -> Array2D:
    """Creates a matrix subjacobian representation from perturbed solutions
    and known perturbation values.

    Parameters
    ----------
    nominal: Array1D
        The solution in nominal conditions.
    perturbed_solutions: Iterable[Array1D]
        The solutions of the perturbed systems
    perturbations: Array1D
        The perturbations that yield these solutions.
    length: int | None
        The length of the nominal solution. Used for cases when the nominal
        solution does not converge.

    """
    # noinspection PyPep8Naming
    J = np.empty((len(nominal) if length is None else length, len(perturbations)))
    for j, (sol, h) in enumerate(zip(perturbed_solutions, perturbations,
                                     strict=True)):
        J[:, j] = (sol - nominal) / h
    return J


def _uq_single_jacobian(j, sys, stat) -> Uncertuple:
    return np.abs(j @ sys), np.sqrt((j ** 2) @ (stat ** 2))


def _uncert_add(x: Uncertuple, y: Uncertuple) -> Uncertuple:
    return x[0] + y[0], np.sqrt(x[1] ** 2 + y[1] ** 2)


_d_as_matrix = delayed(_as_matrix, pure=True)
_d_uq_single_jacobian = delayed(_uq_single_jacobian, pure=True, nout=2)
_d_uncert_add = delayed(_uncert_add, pure=True, nout=2)


class _UQModel:
    """Parent class for UQ models that are cached.

    Used for code reuse, not as a supertype.
    Contains getter and setter functions to ensure cache invalidation and that
    the nominal solution is up-to-date.

    """

    def __init__(self, parameters, model, nominal, step_strategy):
        self._parameters = parameters
        self._model = model
        self.nominal = nominal
        self._step_strategy = step_strategy
        self._cache = {}

    @property
    def model(self) -> Model:
        """The Model of the problem, which takes parameter values and returns a
        DataFrame.

        """
        return self._model

    @model.setter
    def model(self, mod: Model) -> None:
        self._model = mod
        self.nominal = self._eval()
        self._invalidate_cache()

    @property
    def parameters(self) -> dict[str, Value]:
        """The nominal model parameters. Must match the parameters of the model.

        """
        return self._parameters

    @parameters.setter
    def parameters(self, params: dict[str, Value]) -> None:
        self._parameters = params
        self.nominal = self._eval()
        self._invalidate_cache()

    @property
    def step_strategy(self) -> Callable[[T], T]:
        """A strategy for deciding how big a perturbation step to take.
        This should be a small positive value, usually.

        """
        return self._step_strategy

    @step_strategy.setter
    def step_strategy(self, st: Callable[[T], T]) -> None:
        self._step_strategy = st
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        self._cache = {}

    def _eval(self, **parameters) -> DataFrame:
        return self.model(**(self.parameters | parameters))


class UQModel(_UQModel):
    """This object performs the Jacobian analysis for numerical evaluation of
    the solution values against the model parameters.

    """

    def __init__(self, parameters: dict[str, Value],
                 model: Model,
                 nominal: DataFrame = None,
                 step_strategy: Callable[[T], T] = default_deriv_step):
        """
        Parameters
        ----------
        parameters: dict[str, Value]
            The nominal model parameters. Must match the parameters of the model.
        model: Model
            The Model of the problem, which takes parameter values and returns a
            DataFrame.
        nominal: DataFrame
            The result of running the model on the default parameters. Defaults
            to running it on instantiation.
        step_strategy: Callable[[Value], Array1D]
            A strategy for deciding how big a perturbation step to take.
            This should be a small positive value, usually.
        """
        super().__init__(parameters, model, nominal, step_strategy)
        self.nominal = nominal if nominal is not None else model(**parameters)

    def __len__(self) -> int:
        return len(self.nominal.value.values)

    def evaluate(self, **parameters) -> Array1D:
        """Evaluate the solver with different parameter values.

        """
        return self.model(**(self.parameters | parameters)).value.values

    def subjacobian(self, param: str) -> Array2D:
        r"""Derive the Jacobian matrix of derivation by a single input parameter.

        The Jacobian is computed through the simplest finite differences scheme,
        where schematically:

        .. math::
            J_{ij}=\frac{df_i}{dx_j}=\frac{f_i(\mathbf{x_0}+h_j)
                   -f_i(\mathbf{x_0})}{h_j}+\mathcal{O}(h_j)


        Where :math:`\mathbf{x_0} + h_j` means only the :math:`j`-th index in
        :math:`\mathbf{x_0}` is adjusted by :math:`h_j`.


        Parameters
        ----------
        param: str
            Parameter to perform the derivation by. Must be in parameters.

        Returns
        -------
        Array2D
            A 2D matrix of the Jacobian submatrix for the output in regard to the given
            parameter

        """
        try:
            return self._cache[param]
        except KeyError:
            pass
        x = self.parameters[param]
        h = self.step_strategy(x)
        sols = [self.evaluate(**{param: v}) for v in _vector_step(x, h)]
        result = _as_matrix(self.nominal.value.values, sols, np.atleast_1d(h))
        self._cache[param] = result
        return result

    def _uq_single(self, param: str, uncertainty: Uncertainty
                   ) -> Uncertuple:
        """Perform the uncertainty quantification of the model for one
        parameter's uncertainty.

        Parameters
        ----------
        param: str
            The parameter whose uncertainty to consider.
        uncertainty: Uncertainty
            The uncertainty of that parameter.

        Returns
        -------
        Value, Value
            The systematic and statistical uncertainties in each output field.

        """
        j = self.subjacobian(param)
        x = np.atleast_1d(self.parameters[param])
        sys_uncer, stat_uncer = uncertainty.as_absolute(x)
        return _uq_single_jacobian(j, sys_uncer, stat_uncer)

    def uq(self, **uncertainties: Uncertainty) -> Uncertuple:
        """Performs the uncertainty quantification of the model for a given set
        of uncertainties.
        Assumes the uncertainties of different parameters are independent.

        Parameters
        ----------
        uncertainties: Uncertainty
            The uncertainties in each parameter to include in the analysis.
            Keys must be names of parameters of the model.
            Any parameter in the model that isn't listed is considered to have
            zero uncertainty.

        Returns
        -------
        tuple[Value, Value]
            The systematic and statistical uncertainty in each output parameter for the
            given input uncertainties.

        """
        uq_values = (self._uq_single(key, uncertainty)
                     for key, uncertainty in uncertainties.items())
        return reduce(_uncert_add, uq_values, (0, 0))

    def uq_attach(self, **uncertainties: Uncertainty) -> DataFrame:
        """Performs the uncertainty quantification of the model for a given set
        of uncertainties, just like :meth:`uq`.

        Unlike :meth:`uq`, it returns the nominal model output DataFrame where
        two columns are added: **sys** (systematic uncertainty) and
        **stat** (statistical uncertainty).

        Parameters
        ----------
        uncertainties: Uncertainty
            The uncertainties in each parameter to include in the analysis.
            Keys must be names of parameters of the model.
            Any parameter in the model that isn't listed is considered to have
            zero uncertainty.

        Returns
        -------
        DataFrame
            Nominal DataFrame with added uncertainty columns.
        """
        sys, stat = self.uq(**uncertainties)
        return self.nominal.assign(sys=sys, stat=stat)


@delayed(pure=True)
def _join(*dfs):
    return concat(dfs, ignore_index=True, copy=False)


class DASKUQModel(_UQModel):
    """A UQ model which is computed asynchronously, using DASK.

    """

    def __init__(self, parameters: dict[str, Value],
                 model: Model,
                 *features: DelayedModel,
                 feature_length: int = None,
                 persist: bool = False,
                 step_strategy: Callable[[T], T] = default_deriv_step):
        """

        The functions given to this object for the model creation and model features
        should be ``delayed`` in advance, and where possible, mark them as ``pure`` and
        with a correct ``nout``.
        The reason we don't just do this ourselves is that we want a mapping of
        DASKUQModels that use the same function but with a different set of parameters
        to realize that they are actually the same function.

        It is also preferred to let people do their own delaying, so we don't get an
        accidental double-delay.

        Parameters
        ----------
        parameters: dict[str, Value]
            The nominal model parameters. Must match the parameters of the
            features and model factory.
        model: Model
            The Model of the problem, which takes parameter values and returns a
            DataFrame.
        features: DelayedModel
            Functions that take a model and the set of creation parameters and
            return DataFrames with the format used by the Aggregator
            (have calculation, variable, i, j, and so on).
            Rows from different DataFrames given by features should be distinct,
            meaning that they should have a different value somewhere that isn't the
            "value" column.
        persist: bool
            Whether values should persist by default. Defaults to False.
        feature_length: int
            The length of a nominal solution for all the features.
            Used to set the length of the jacobian when the nominal solution
            does not converge.
        step_strategy: Callable[[Value], Array1D]
            A strategy for deciding how big a perturbation step to take.
            This should be a small positive value, usually.
        """
        super().__init__(parameters, model, None, step_strategy)
        self.features = features
        self.nominal = self._eval()
        self._len = feature_length
        self.persist = persist

    def _eval(self, **parameters) -> Delayed:
        params = self.parameters | parameters
        # noinspection PyArgumentList
        pieces = [feature(self.model, **params) for feature in self.features]
        return _join(*pieces)

    def evaluate(self, **parameters) -> Delayed:
        """Evaluate the solver with different parameter values.

        """
        df = self._eval(**parameters)
        return df.value.values

    def subjacobian(self, param: str, persist: bool = None) -> Delayed:
        r"""Derive the Jacobian matrix of derivation by a single input parameter.

        See Also
        --------
        :meth:`UQModel.subjacobian`

        Parameters
        ----------
        param: str
            Parameter to perform the derivation by. Must be in parameters.
        persist: bool
            Flag for whether to persist the result or not (caching). Defaults to the attribute value of this object.

        Returns
        -------
        Array2D
            A 2D matrix of the Jacobian submatrix for the output in regard to the given
            parameter

        """
        try:
            return self._cache[param]
        except KeyError:
            pass
        x = self.parameters[param]
        h = self.step_strategy(x)
        sols = [self.evaluate(**{param: v}) for v in _vector_step(x, h)]
        result = _d_as_matrix(self.nominal.value.values, sols, np.atleast_1d(h),
                              length=self._len)
        persist = self.persist if persist is None else persist
        if persist:
            result = result.persist()
        self._cache[param] = result
        return result

    def _uq_single(self, param: str, uncertainty: Uncertainty, persist: bool = None,
                   ) -> tuple[Delayed, Delayed]:
        """Perform the uncertainty quantification of the model for one
        parameter's uncertainty.

        Parameters
        ----------
        param: str
            The parameter whose uncertainty to consider.
        uncertainty: Uncertainty
            The uncertainty of that parameter.
        persist: bool
            Flag for whether to persist the result or not (caching).

        Returns
        -------
        Value, Value
            The systematic and statistical uncertainties in each output field.

        """
        j = self.subjacobian(param, persist=persist)
        x = np.atleast_1d(self.parameters[param])
        sys_uncer, stat_uncer = uncertainty.as_absolute(x)
        return _d_uq_single_jacobian(j, sys_uncer, stat_uncer)

    def uq(self, persist: bool = None, **uncertainties: Uncertainty,
           ) -> tuple[Delayed, Delayed]:
        """Performs the uncertainty quantification of the model for a given set
        of uncertainties.
        Assumes the uncertainties of different parameters are independent.

        Parameters
        ----------
        persist: bool
            Flag for whether to persist the result or not (caching).
        uncertainties: Uncertainty
            The uncertainties in each parameter to include in the analysis.
            Keys must be names of parameters of the model.
            Any parameter in the model that isn't listed is considered to have
            zero uncertainty.

        Returns
        -------
        tuple[Value, Value]
            The systematic and statistical uncertainty in each output parameter for the
            given input uncertainties.

        """
        if not uncertainties:
            zd = delayed(0, pure=True)
            return zd, zd
        uq_values = (self._uq_single(key, uncertainty, persist=persist)
                     for key, uncertainty in uncertainties.items())
        sys, stat = reduce(_d_uncert_add, uq_values, (0, 0))
        # Safe because if the iterator is empty and the initial values persist,
        # we would have returned early with delayed values.
        # noinspection PyTypeChecker
        return sys, stat

    def uq_attach(self, persist: bool = None, **uncertainties: Uncertainty) -> Delayed:
        """Performs the uncertainty quantification of the model for a given set
        of uncertainties, just like :meth:`uq`.

        Unlike :meth:`uq`, it returns the nominal model output DataFrame where
        two columns are added: **sys** (systematic uncertainty) and
        **stat** (statistical uncertainty).
        Since this is a DaskUQModel, this is a delayed of said operation.

        Parameters
        ----------
        uncertainties: Uncertainty
            The uncertainties in each parameter to include in the analysis.
            Keys must be names of parameters of the model.
            Any parameter in the model that isn't listed is considered to have
            zero uncertainty.

        Returns
        -------
        Delayed
            Nominal DataFrame with added uncertainty columns.
        """
        sys, stat = self.uq(persist=persist, **uncertainties)
        return self.nominal.assign(sys=sys, stat=stat)
