from dataclasses import asdict, dataclass

import numpy as np
from cytoolz import valfilter

from stream.units import Value

Uncertuple = tuple[Value, Value]


@dataclass(frozen=True)
class Uncertainty:
    r"""Describes perturbations, both statistical and systematic.
    In relation to a given variable :math:`x` which, unperturbed, is denoted :math:`x_0`.
    The perturbed value is assumed to have the following distribution :math:`X`:

    .. math::
        X \sim x_0 (1 + \mu_\text{sys, rel.}) \mathcal{N}(1, \sigma^2_\text{stat, rel.})
        + \mathcal{N}(\mu_\text{sys, abs.}, \sigma^2_\text{stat, abs.})

    Where :math:`\mu_\text{sys, rel.}` defines a systematic relative uncertainty,
    :math:`\sigma_\text{stat, rel.}` defines a statistical relative uncertainty,
    :math:`\mu_\text{sys, abs.}` defines a systematic absolute uncertainty and
    :math:`\sigma_\text{stat, abs.}` defines statistical absolute uncertainty
    """

    systematic_relative: Value = 0.0
    statistical_relative: Value = 0.0
    systematic_absolute: Value = 0.0
    statistical_absolute: Value = 0.0

    def as_absolute(self, nominal: Value) -> Uncertuple:
        """Returns an absolute representation of the systematic and statistical
        uncertainties.

        Parameters
        ----------
        nominal: Value
            The nominal value to use for the relative-to-absolute conversion.

        """
        return (
            self.systematic_relative * nominal + self.systematic_absolute,
            np.sqrt((self.statistical_relative * nominal) ** 2 + self.statistical_absolute**2),
        )

    def __bool__(self):
        return any(
            (
                self.systematic_relative,
                self.statistical_relative,
                self.statistical_absolute,
                self.systematic_absolute,
            )
        )

    def __add__(self, other: "Uncertainty") -> "Uncertainty":
        if not isinstance(other, Uncertainty):
            return NotImplemented
        return Uncertainty(
            self.systematic_relative + other.systematic_relative,
            np.sqrt(self.statistical_relative**2 + other.statistical_relative**2),
            self.systematic_absolute + other.systematic_absolute,
            np.sqrt(self.statistical_absolute**2 + other.statistical_absolute**2),
        )

    def __repr__(self) -> str:
        non_zeros = valfilter(np.any, asdict(self))
        val_string = ", ".join([f"{k}={v}" for k, v in non_zeros.items()])
        return f"Uncertainty({val_string})"
