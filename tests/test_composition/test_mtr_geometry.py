import numpy as np

from stream.composition.mtr_geometry import chain_fuels_channels
from stream.aggregator import vars_


def test_chaining_one_channel_more_than_fuels_has_correct_graph_for_specific_case():
    r"""testing :func:`~stream.composition.mtr_geometry.chain_fuels_channels`
    behavior"""
    channels = np.arange(0, 10)
    fuels = "ABCDEFGHI"
    assert len(channels) - len(fuels) == 1

    rod_ = chain_fuels_channels(channels=channels, fuels=fuels)
    assert len(rod_.graph) == len(channels) + len(fuels)
    assert list(rod_.graph.nodes) == [0, 'A', 1, 'B', 2, 'C', 3, 'D', 4,
                                      'E', 5, 'F', 6, 'G', 7, 'H', 8, 'I', 9]
    assert list(rod_.graph.subgraph(channels).edges) == []
    assert list(rod_.graph.subgraph(fuels).edges) == []
    assert (list(rod_.graph.edges(2, data=True))
            == [(2, 'B', vars_('T_right', 'h_right')),
                (2, 'C', vars_('T_left', 'h_left'))])
    assert (list(rod_.graph.edges('A', data=True))
            == [('A', 0, vars_('T_right')),
                ('A', 1, vars_('T_left'))])


def test_chaining_one_fuel_more_than_channels_has_correct_graph_for_specific_case():
    r"""testing :func:`~stream.composition.mtr_geometry.chain_fuels_channels`
    behavior"""
    channels = np.arange(8)
    # noinspection SpellCheckingInspection
    fuels = "ABCDEFGHI"
    assert len(channels) - len(fuels) == -1
    rod_ = chain_fuels_channels(channels=channels, fuels=fuels)
    assert len(rod_.graph) == len(channels) + len(fuels)
    assert list(rod_.graph.nodes) == ['A', 0, 'B', 1, 'C', 2, 'D', 3, 'E', 4,
                                      'F', 5, 'G', 6, 'H', 7, 'I']
    assert list(rod_.graph.subgraph(channels).edges) == []
    assert list(rod_.graph.subgraph(fuels).edges) == []
    assert (list(rod_.graph.edges(2, data=True))
            == [(2, 'D', vars_('T_left', 'h_left')),
                (2, 'C', vars_('T_right', 'h_right'))])
    assert (list(rod_.graph.edges('B', data=True))
            == [('B', 0, vars_('T_right', )),
                ('B', 1, vars_('T_left', ))])
