from artemis.general.redict import ReDict, MultipleMatchesError, ReCurseDict
import pytest

__author__ = 'peter'


def test_redict():

    d = ReDict({
        'alpha-.*': 1,
        'alpha_.*': 2,
        'beta-.*': 3,
        '.*-gamma': 4,
        'eps.*': 5,
        'phi-(eta|pi)': 6,
        'phi-zeta': 7,
        'phi-omega$': 7
        })

    assert d['alpha-beta'] == 1
    assert d['alpha_beta'] == 2
    with pytest.raises(MultipleMatchesError):
        d['alpha-gamma']  # Multiple Matches
    assert d.get_matches('alpha-gamma') == {'alpha-.*': 1, '.*-gamma': 4}
    assert d['alpha-delta'] == 1
    with pytest.raises(KeyError):
        d['delta-alpha']  # No match
    with pytest.raises(KeyError):
        d['delta_gamma']  # No match
    assert d.get_matches('delta_gamma') == {}
    assert d.get_matches('delta-alpha') == {}
    assert d['beta-delta'] == 3
    assert d['delta-gamma'] == 4
    assert d['epsilon-phi'] == 5
    assert d['phi-eta'] == 6
    assert d['phi-pi'] == 6
    assert d['phi-zeta'] == 7
    with pytest.raises(KeyError):
        d['phi-eat']  # Which would not be the case if we used 'phi-[eta|pi]' (square brackets)
    with pytest.raises(KeyError):
        d['phi-psi']
    assert d['phi-zeta-something'] == 7  # To be aware of -
    with pytest.raises(KeyError):
        d['phi-omega-something']

def test_default_handling():

    # Option 1: The usual dict method
    d = ReDict({
        'alpha-.*': 1,
        'alpha_.*': 2,
        'beta-.*': 3,
        '.*-gamma': 4,
        'eps.*': 5
        })

    assert d.get('alpha-beta', 6) == 1
    assert d.get('delta-alpha', 6) == 6
    with pytest.raises(MultipleMatchesError):
        d.get('alpha-gamma', 6)  # Multiple Matches

    # Option 2 - This is useful in ReCurseDicts where we want multiple levels of defaults.
    d = ReDict({
        'alpha-.*': 1,
        'alpha_.*': 2,
        'beta-.*': 3,
        '.*-gamma': 4,
        'eps.*': 5,
        None: 6
        })

    assert d['alpha-beta'] == 1
    assert d['delta-alpha'] == 6
    with pytest.raises(MultipleMatchesError):
        d['alpha-gamma']  # Multiple Matches


def test_recurse_dict():

    d = ReCurseDict({
        'alpha-*': {
            '.*-een': 1,
            '.*-twee': 2,
            '.*-drie': 3,
            None: -1
            },
        'beta-*': 4,
        'gamma-*': {
            'forgot_prefix': 5,
            '.*remembered_prefix': 6
            },
        None: -2,
    })

    assert d['alpha-een'] == 1
    assert d['alpha-twee'] == 2
    assert d['alpha-vier'] == -1
    assert d['beta-anything'] == 4
    with pytest.raises(KeyError):
        assert d['gamma-forgot_prefix']
    assert d['gamma-remembered_prefix'] == 6
    assert d['delta-anything'] == -2


if __name__ == '__main__':
    test_recurse_dict()
    test_default_handling()
    test_redict()
