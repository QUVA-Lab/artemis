from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from builtins import input
from builtins import zip
import inspect
import shlex
from collections import OrderedDict

# Code taken and modified from:
# Taken from https://github.com/braincorp/dead_easy_ui/


class DeadEasyUI(object):
    """
    A tool for making a small program that can be run either programmatically or as a command-line user interface.  Use
    this class by extending it.  Whatever methods you add to the extended class will become commands that you can run.

    Example (which you can also run at the bottom of this file):

        class MyUserInterface(DeadEasyUI):

            def isprime(self, number):
                print '{} is {}prime'.format(number, {False: '', True: 'not '}[next(iter([True for i in range(3, number) if number%i==0]+[False]))])

            def showargs(self, arg1, arg2):
                print '  arg1: {}: {}\n  arg2: {}: {}\n'.format(arg1, type(arg1), arg2, type(arg2))

        MyUserInterface().launch(run_in_loop=True)

    This will bring up a UI

        ==== MyUserInterface console menu ====
        Command Options: {isprime, showargs}
          Enter Command or "help" for help >> isprime 117
        117 is not prime
        ==== MyUserInterface console menu ====
        Command Options: {isprime, showargs}
          Enter Command or "help" for help >> showargs 'abc' arg2=4.3
          arg1: abc: <type 'str'>
          arg2: 4.3: <type 'float'>

    """

    def _get_menu_string(self):
        return '==== {} console menu ====\n'.format(self.__class__.__name__) if (self.__doc__ is None or self.__doc__ == DeadEasyUI.__doc__) else self.__doc__ if self.__doc__.endswith('\n') else self.__doc__+ '\n'

    def launch(self, prompt = 'Enter Command or "h" for help >> ', run_in_loop = True, arg_handling_mode='fallback'):
        """
        Launch a command-line UI.
        :param prompt:
        :param run_in_loop:
        :param arg_handling_mode: Can be ('str', 'guess', 'literal')
            'str': Means pass all args to the method as strings
            'literal': Means use eval to parse each arg.
            'fallback': Try literal parsing, and if it fails, fall back to string
        :return:
        """

        def linenumber_of_member(k, m):
            try:
                return m.__func__.__code__.co_firstlineno
            except AttributeError:
                return -1

        mymethods = sorted(inspect.getmembers(self, predicate=inspect.ismethod),
                           key = lambda pair: linenumber_of_member(*pair))
        mymethods = [(method_name, method) for method_name, method in mymethods if method_name!='launch' and not method_name.startswith('_')]
        mymethods = OrderedDict(mymethods)

        options_doc = 'Command Options: {{{}}}'.format(', '.join([k for k in list(mymethods.keys())]+['quit', 'help']))

        skip_info = False
        while True:
            doc = self._get_menu_string()

            if not skip_info:
                print('{}{}'.format(doc, options_doc))
            user_input = input('  {}'.format(prompt))
            cmd, args, kwargs = parse_user_function_call(user_input, arg_handling_mode=arg_handling_mode)

            if cmd is None:
                continue

            skip_info = False
            if cmd in ('h', 'help'):
                print(self._get_help_string(mymethods=mymethods, method_names_for_help=[args[0]] if len(args) > 0 else None))
                skip_info = True
                continue
            elif cmd in ('q', 'quit'):
                print('Quitting {}.  So long.'.format(self.__class__.__name__))
                break
            elif cmd in mymethods:
                mymethods[cmd](*args, **kwargs)
            else:
                print("Unknown command '{}'.  Options are {}".format(cmd, list(mymethods.keys())+['help']))
                skip_info = True
                continue
            if not run_in_loop:
                break

    def _get_help_string(self, mymethods, method_names_for_help=None):
        string = ''
        string += '----------------------------\n'
        string += "To run a command, type the method name and then space-separated arguments.  e.g.\n    >> my_method 1 'string-arg' named_arg=2\n\n"
        if method_names_for_help is None:
            method_names_for_help = list(mymethods.keys())
        if len(method_names_for_help) == 0:
            string+= "Class {} has no methods, and is therefor a useless console menu.  Add methods.\n".format(
                self.__class__.__name__)
        for method_name in method_names_for_help:
            argspec = inspect.getargspec(mymethods[method_name])
            default_start_ix = len(argspec.args) if argspec.defaults is None else len(argspec.args) - len(
                argspec.defaults)
            argstring = ' '.join([repr(a) for a in argspec.args[1:default_start_ix]] +
                ['[{}={}]'.format(a, repr(v)) for a, v in zip(argspec.args[default_start_ix:], argspec.defaults if argspec.defaults is not None else [])]) \
                if len(argspec.args)>1 else '<No arguments>'
            doc = mymethods[method_name].__doc__ if mymethods[method_name].__doc__ is not None else '<No documentation>'
            string+= '- {} {}: {}\n'.format(method_name, argstring, doc)
        string += '----------------------------\n'
        return string


def parse_user_function_call(cmd_str, arg_handling_mode = 'fallback'):
    """
    A simple way to parse a user call to a python function.  The purpose of this is to make it easy for a user
    to specify a python function and the arguments to call it with from the console.  Example:

        parse_user_function_call("my_function 1 'two' a='three'") == ('my_function', (1, 'two'), {'a': 'three'})

    Other code can use this to actually call the function

    Parse arguments to a Python function
    :param str cmd_str: The command string.  e.g. "my_function 1 'two' a='three'"
    :param forgive_unquoted_strings: Allow for unnamed string args to be unquoted.
        e.g. "my_function my_arg_string" would interpreted as "my_function 'my_arg_string' instead of throwing an error"
    :return: The function name, args, kwargs
    :rtype: Tuple[str, Tuple[Any]. Dict[str: Any]
    """

    assert arg_handling_mode in ('str', 'literal', 'fallback')

    # def _fake_func(*args, **kwargs):
    #     Just exists to help with extracting args, kwargs
        # return args, kwargs

    cmd_args = shlex.split(cmd_str, posix=False)
    assert len(cmd_args) == len(shlex.split(cmd_str, posix=True)), "Parse error on string '{}'. You're not allowed having spaces in the values of string keyword args:".format(cmd_str)

    if len(cmd_args)==0:
        return None, None, None

    func_name = cmd_args[0]

    def parse_arg(arg_str):
        if arg_handling_mode=='str':
            return arg_str
        elif arg_handling_mode=='literal':
           return eval(arg_str, {}, {})
        else:
            try:
                return eval(arg_str, {}, {})
            except:
                return arg_str

    args = []
    kwargs = {}
    for arg in cmd_args[1:]:
        if '=' not in arg:  # Positional
            assert len(kwargs)==0, 'You entered a positional arg after a keyword arg.  Keyword args {} aleady exist.'.format(kwargs)
            args.append(parse_arg(arg))
        else:
            arg_name, arg_val = arg.split('=', 1)
            kwargs[arg_name] = parse_arg(arg_val)

    return func_name, tuple(args), kwargs

    # if forgive_unquoted_strings:
    #     cmd_args = [cmd_args[0]] + [_quote_args_that_you_forgot_to_quote(arg) for arg in cmd_args[1:]]
    #
    # args, kwargs = eval('_fake_func(' + ','.join(cmd_args[1:]) + ')', {'_fake_func': _fake_func}, {})
    # return func_name, args, kwargs


def _quote_args_that_you_forgot_to_quote(arg):
    """Wrap the arg in quotes if the user failed to do it."""
    if arg.startswith('"') or arg.startswith("'"):
        return arg
    elif '=' in arg and sum(a=='=' for a in arg)==1:  # Keyword
        name, val = arg.split('=')
        if val[0].isalpha():
            return '{}="{}"'.format(name, val)
        else:
            return arg
    else:
        if arg[0].isalpha():
            return '"{}"'.format(arg)
        else:
            return arg


if __name__ == '__main__':

    class MyUserInterface(DeadEasyUI):

        def isprime(self, number):
            print('{} is {}prime'.format(number, {False: '', True: 'not '}[next(iter([True for i in range(3, number) if number % i==0]+[False]))]))

        def showargs(self, arg1, arg2):
            print('  arg1: {}: {}\n  arg2: {}: {}\n'.format(arg1, type(arg1), arg2, type(arg2)))

    MyUserInterface().launch(run_in_loop=True)
