__author__ = 'peter'


class ReportCollector(object):

    def __init__(self, display = True):
        self.display = display
        self._report = []

    def append(self, update):
        self._report.append(update)
        if self.display:
            print(update)

    def get_report(self):
        return self._report

    def get_report_text(self):
        return '\n'.join(self._report)

    def print_report(self):
        print('='*15+' Report '+'='*15+'\n'+self.get_report_text()+'\n'+'='*40+'\n')
