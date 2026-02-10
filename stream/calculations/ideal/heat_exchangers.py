from stream.calculations.ideal.ideal import LumpedComponent
from stream.units import Celsius

__all__ = ["HeatExchanger"]


class HeatExchanger(LumpedComponent):
    """This heat exchanger doesn't care about input temperature, it always
    returns the same temperature. Additionally, it exerts no pressure
    difference.
    """

    def __init__(self, outlet: Celsius, name: str = "HX"):
        self.name = name
        self.T = outlet

    def T_out(self, **_) -> Celsius: return self.T
