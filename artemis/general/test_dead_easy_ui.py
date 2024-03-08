from artemis.general.dead_easy_ui import parse_user_function_call


def test_parse_user_function_call():

    assert parse_user_function_call("myfunc 1 a 'a b' c=3 ddd=[2,3] ee='abc'") == ("myfunc", (1, 'a', 'a b'), dict(c=3, ddd=[2, 3], ee='abc'))


if __name__ == '__main__':
    test_parse_user_function_call()
