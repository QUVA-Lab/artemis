from artemis.fileman.primitive_specifiers import PrimativeSpecifier, load_primative

__author__ = 'peter'

def test_primative_specifiers():

    class MyThing(PrimativeSpecifier):

        def __init__(self, a, b, c):
            self.a=a
            self.b=b
            self.c=c

    thing1 = MyThing(1, 2, 3)

    primative = thing1.to_primative()

    thing2 = load_primative(primative)
    assert thing2.a==1
    assert thing2.b==2
    assert thing2.c==3


if __name__ == '__main__':
    test_primative_specifiers()
