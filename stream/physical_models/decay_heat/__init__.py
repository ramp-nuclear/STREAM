r"""
In addition to "instantaneous" power generated from nuclear fission,
other forms of nuclear decay contribute to power generated in the reactor core.
These include the delayed neutron-induced fission, decay of fission products and actinides, and activation of materials.
The first is dealt with by the Point Kinetics model (see :class:`~.PointKinetics`)
"""

from stream.physical_models.decay_heat import actinides, activation, fission_products, fissions

__all__ = [
    "actinides",
    "activation",
    "fissions",
    "fission_products",
]
