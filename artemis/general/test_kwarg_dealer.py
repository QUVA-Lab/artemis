from artemis.general.kwarg_dealer import KwargDealer
import pytest

__author__ = 'peter'


def my_favourite_system(planet, **kwargs):
    kd = KwargDealer(kwargs)

    if planet == 'Earth':
        params = kd.deal({'moon': 'The Moon', 'person': 'Jesus'})
        favsys = 'My favourite planet is %s, my favourite moon is %s, and my favourite person is %s' %(planet, params['moon'], params['person'])
    elif planet == 'Mars':
        params = kd.deal({'moon': 'Phobos'})
        favsys = 'My favourite planet is %s, my favourite moon is %s' % (planet, params['moon'])
    elif planet == 'Venus':
        favsys = 'My favourite planet is %s.' % (planet, )

    kd.assert_empty()
    return favsys


def test_kwarg_dealer():

    assert my_favourite_system('Earth') == 'My favourite planet is Earth, my favourite moon is The Moon, and my favourite person is Jesus'
    assert my_favourite_system('Earth', person = 'Margerit Thatcher') == 'My favourite planet is Earth, my favourite moon is The Moon, and my favourite person is Margerit Thatcher'
    assert my_favourite_system('Mars') == 'My favourite planet is Mars, my favourite moon is Phobos'
    assert my_favourite_system('Venus') == 'My favourite planet is Venus.'

    with pytest.raises(AssertionError):
        my_favourite_system('Mars', person = 'Jesus')  # No people on Mars

    with pytest.raises(AssertionError):
        my_favourite_system('Venus', moon = 'Europa')  # No moons around Venus

if __name__ == '__main__':
    test_kwarg_dealer()
