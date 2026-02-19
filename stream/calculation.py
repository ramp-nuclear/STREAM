r"""
A module defining what is a :class:`Calculation`, in the sense of a first order in time
differential and algebraic system of equation. A :class:`Calculation` is a subset of
such a system.
"""
from abc import abstractmethod
from functools import wraps
from typing import Any, Optional, Protocol, runtime_checkable, Sequence, Iterable

import numpy as np
from cytoolz.dicttoolz import valmap

from stream.units import Array1D, Name, Place, Value
from stream.utilities import flatten_values


__all__ = ["Calculation", "unpacked", 'CalcState']
CalcState = dict[Name, Value]


@runtime_checkable
class Calculation(Protocol):
    r"""
    A calculation in the context of STREAM is derived from the following DAE,
    which is a first order in time differential and algebraic equation:
    :math:`M\frac{d\vec{y}}{dt} = \vec{F}\left(\vec{y}, t\right)`

    Where :math:`M_{ii}=1,0` is the matrix defining whether
    :math:`F_i(\vec{y}, t)` is the
    differential or an algebraic equation :math:`F_i(\vec{y},t) = 0`.
    :math:`\vec{y}` is the variable vector, from which a subset is the input
    for each individual calculation. The calculation yields a subset
    of :math:`F(\vec{y}, t)`, for which it is deemed the unique owner.

    Therefore,

    .. note::
        One may say that a Calculation
        **is** a subset of :math:`\vec{F} \left(\vec{y}, t\right)`

    **Three kinds of inputs are distinguished:**

    1. **Model Parameters**: Attributes of the calculation itself,
    independent of all other inputs. Supposedly, these are component and
    physical constants.

    2. **Input Functions**: These are functions through which a user may
    interfere with a system. Namely, they are only time-dependent. A
    calculation should not be made aware of the separation between them and
    Variables.

    3. **Variables**: These are the main input for a calculation. Every
    property having distinct time dependence which is not user defined is a
    variable.
    """

    name: str

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__

    @abstractmethod
    def calculate(self, variables: Sequence[float], **_) -> Array1D:
        """The main method of a calculation

        Parameters
        ----------
        variables: Sequence[float]
            the variables required for the calculation,
            which the calculation handles itself.


        Requested external variables can be passed via keyword arguments.

        Returns
        -------
        Output results: Array1D
            subset of the differential/algebraic functional result.
        """
        raise NotImplementedError

    def indices(
            self, variable: Name, asking: Optional["Calculation"] = None
            ) -> Place | dict[Name, Place]:
        """For a given variable name, return the appropriate positions in the vector

        Parameters
        ----------
        variable: Name
            Name of requested variable
        asking: Calculation or None
            What calculation is asking for the indices?
            For example, this is important in
            :class:`~.stream.calculations.kirchhoff.Kirchhoff`.

        Returns
        -------
        indices: Place or dict[Calculation, Place]
            The place in which the calculation uses the variable, or a
            dictionary with variable names related to this name and
            their places.
        """
        return self.variables[variable]

    @property
    @abstractmethod
    def mass_vector(self) -> Sequence[bool]:
        r"""Each entry corresponds to an equation. For each entry the value is
        either 0 or 1:

        0) Algebraic equation :math:`F(\vec{y},t)=0`

        1) Differential equation :math:`\dot{y}=F(\vec{y},t)`

        Returns
        -------
        mass: Sequence[bool]
         The "mass" of each variable under this calculation's control.
         Values are 0 for algebraically defined terms and 1 for differentially
         defined terms.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of slots used by the calculation, meaning length of
        calculate()'s output"""
        return len(self.mass_vector)

    @property
    @abstractmethod
    def variables(self) -> dict[str, Place]:
        """All variables owned by calculation"""
        raise NotImplementedError

    def load(self, state: CalcState) -> Array1D:
        """
        Given a state of the calculation, return the untagged corresponding
        array, which may be used to calculate the next step.

        The default method implemented here assumes the state is completely
        described by the variables presented in self.variables.

        Parameters
        ----------
        state: CalcState
            Tagged information regarding system state

        Returns
        -------
        y: Array1D
            Calculation ready array
        """
        y = np.empty(len(self))
        for var, place in self.variables.items():
            y[place] = state[var]
        return y

    def save(self, vector: Sequence[float], **_) -> CalcState:
        """
        Given input for "calculate" (which is a legal state of the system),
        tag the information, i.e. create a "State" and return it.

        The default method implemented here assumes the state is completely
        described by the variables presented in self.variables.
        This may not be the best descriptor of the state, as there can be more
        favourable expressions. This is the separation between ``strict_save``, which
        is the vanilla version described above, and ``save``, which should be free to
        add other keys to the "State".

        Parameters
        ----------
        vector: Sequence[float]
            Input

        Returns
        -------
        state: CalcState
            Tagged information regarding system state
        """
        return valmap(vector.__getitem__, self.variables)

    strict_save = save

    def should_continue(self, variables: Sequence[float], **_) -> bool:
        r"""
        A function to be used in transient calculations to determine stopping
        conditions at physical states of a system. Returning False signals the
        calculation is to be stopped. Logging the reason is encouraged.

        This function's inputs should equal those of :meth:`save` and :meth:`calculate`.

        .. note::
            Implementation in :class:`~stream.aggregator.Aggregator` dictates
            the :meth:`change_state` method is called right before :meth:`should_continue`,
            with the same input.

        Parameters
        ----------
        variables: Sequence[float]
            Input

        Returns
        -------
        should_continue: bool
            Should the transient simulation continue?
        """
        return True

    def change_state(self, variables: Sequence[float], **_):
        r"""
        A function to be used in transient calculations called at physical states
        of a system. If the calculation has an internal state which depends
        on the physical state, this state should be changed here.

        This function's inputs should equal those of :meth:`save` and :meth:`calculate`.

        Parameters
        ----------
        variables: Sequence[float]
            Input
        """
        pass


def unpacked(calculate=None, *, exclude: Iterable[str] = ()):
    """
    This is a decorator for Calculation methods (calculate and save, mostly),
    to be applied when the origin of the external variables is unimportant.

    Parameters
    ----------
    calculate: callable
        Calculation.calculate actualized method
    exclude: Iterable[str]
        Which variables to exclude from unpacking

    Returns
    -------
    calculate*: callable
        The method which now receives its keyword arguments as
        np.arrays or as floats and not a dictionary
    """

    
    def _unpacked(_calculate):
        @wraps(_calculate)
        def _unpack(*args, **kwargs):
            try:
                excluded_kwargs = {k: kwargs.pop(k) for k in exclude}
                return _calculate(*args, **valmap(_concat, kwargs) | excluded_kwargs)
            except KeyError as e:
                raise KeyError(f"While unpacking in {_calculate}, 'exclude' got a variable name which was not recieved by {_calculate}. Variable name: {e}")
            except BaseException as e:
                e.args = (f"Error found at {_calculate}: {e.args[0]}", *e.args[1:])
                raise

        return _unpack
    
    if calculate is not None:
        # Used as @unpacked
        return _unpacked(calculate)
    else:
        # Used as @unpacked(...)
        return _unpacked


def _concat(v: Value | dict[Any, Value]) -> Array1D:
    try:
        return flatten_values(v)
    except AttributeError:
        return v
