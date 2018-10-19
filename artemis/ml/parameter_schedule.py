import numpy as np


class ParameterSchedule(object):

    def __init__(self, schedule, print_variable_name = None):
        """
        Given a schedule for a changing parameter (e.g. learning rate) get the values for this parameter at a given time.
        e.g.:
            learning_rate_scheduler = ParameterSchedule({0: 0.1, 10: 0.01, 100: 0.001}, print_variable_name='eta')
            new_learning_rate = learning_rate_scheduler.get_new_value(epoch=14)
            assert new_learning_rate == 0.01

        :param schedule: Can be:
            - A dict<epoch: value> where the epoch is a number indicating the training progress and the value
              indicates the value that the parameter should take.
            - A function which takes the epoch and returns a parameter value.
            - A number or array, in which case the value remains constant
        """
        if isinstance(schedule, (int, float, np.ndarray)):
            schedule = {0: schedule}
        if isinstance(schedule, dict):
            assert all(isinstance(num, (int, float)) for num in schedule.keys())
            self._reverse_sorted_schedule_checkpoints = sorted(schedule.keys(), reverse=True)
        else:
            assert callable(schedule)
        self.schedule = schedule
        self.print_variable_name = print_variable_name
        self.last_value = None  # Just used for print statements.

    def __call__(self, epoch):
        if isinstance(self.schedule, dict):
            new_value = self.schedule[next(e for e in self._reverse_sorted_schedule_checkpoints if e <= epoch)]
        else:
            new_value = self.schedule(epoch)
        if self.last_value != new_value and self.print_variable_name is not None:
            print('Epoch {}: {} = {}'.format(epoch, self.print_variable_name, new_value))
            self.last_value = new_value
        return new_value
