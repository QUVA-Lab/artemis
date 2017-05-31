from abc import abstractmethod

__author__ = 'peter'
from copy import deepcopy
import pickle
"""
This module is made to help create "PrimativeSpecifiers".  A Primative Specifier is an object that can be turned
into a primative data structure, independent of all code.  You might want to do this, for instance, if you want to save
a trained model, but you want to be able to load it in any environment (if your code changes, or someone else wants to
load your model into their code.).
"""

_SAVEABLE_CLASSES = {}


class Serializable(object):

    @abstractmethod
    def to_spec(self):
        """
        :return: An object that can be used to reconstruct this class.
        """

    @classmethod
    @abstractmethod
    def from_spec(cls, spec):
        """
        :param spec: An object of the form returned by to_spec.
        :return: An instance of the class
        """


def _register_saveable_class(saveable_class):
    assert issubclass(saveable_class, PrimativeSpecifier)
    class_name = saveable_class.__name__
    if class_name in _SAVEABLE_CLASSES:
        assert saveable_class is _SAVEABLE_CLASSES[class_name], 'Another object has already registered as "%s"' % (saveable_class, )
    _SAVEABLE_CLASSES[class_name] = saveable_class


def load_primative(primative):
    assert '__class__' in primative, 'Saveable object must have a __class__ attribute'
    assert primative['__class__'] in _SAVEABLE_CLASSES, 'No class by the name of "%s" is registered as saveable' % (primative['__class__'])
    d = primative.copy()
    del d['__class__']
    obj = object.__new__(_SAVEABLE_CLASSES[primative['__class__']])
    obj.__dict__.update(d)
    return obj


class PrimativeSpecifier(object):

    def __new__(cls, *args, **kwargs):
        _register_saveable_class(cls)
        return object.__new__(cls)

    def to_primative(self):
        obj_dict = {'__class__': self.__class__.__name__}
        assert '__class__' not in self.__dict__, 'The key "__class__ is reserved, and should not be used by object specifiers.'
        obj_dict.update(self.__dict__)
        return obj_dict

    def clone(self):
        return load_primative(deepcopy(self.to_primative()))

