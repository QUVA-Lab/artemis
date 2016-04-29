import sys
__author__ = 'peter'


class TextWrappingPrinter(object):

    def __init__(self, newline_every = 70):

        self.newline_every = newline_every
        self._space_to_newline = newline_every

    def write(self, text):

        while True:
            try:
                first_newline_index = text.index('\n')
                if first_newline_index < self._space_to_newline:
                    sys.stdout.write(text[:first_newline_index+1])
                    text = text[first_newline_index+1:]
                    self._space_to_newline = self.newline_every
                else:
                    # Need to insert newline
                    sys.stdout.write(text[:self._space_to_newline] + '\n')
                    text = text[self._space_to_newline:]
                    self._space_to_newline = self.newline_every
            except ValueError:  # No newlines
                if len(text) < self._space_to_newline:  # Can put all text into line
                    sys.stdout.write(text)
                    self._space_to_newline -= len(text)
                    break
                else:
                    # Need to insert newline
                    sys.stdout.write(text[:self._space_to_newline] + '\n')
                    text = text[self._space_to_newline:]
                    self._space_to_newline = self.newline_every
