__author__ = 'peter'


class CheckPointCounter(object):
    """
    Scenerio: You have a loop, and want to do something periodically within that loop, but not
    on every iteration.  Use this object to tell you if you want to do that thing or not.

    You give it a series of checkpoints, and each time it is called, it will return the number
    of checkpoints it has passed since it was last called.  If you only want to know IF it's passed
    a checkpoint, you can treat the return as a boolean.
    """

    def __init__(self, checkpoints):
        self._checkpoints = checkpoints
        self._index = 0

    def check(self, progress):
        """
        :param progress: Indicator of the progress - on the same scale of the checkpoints.
        :return: (points_passed, done), where:
            points_passed is the number of checkpoints passed since the last call to this method,
            done is a boolean indicating whether the last checkpoint has been passed.

        Note that done will only be True of points_passed>0
        """
        counter = 0
        done = False

        while True:
            if self._index == len(self._checkpoints):
                done = True
                break
            elif progress < self._checkpoints[self._index]:
                break
            else:
                counter += 1
                self._index += 1

        return counter, done
