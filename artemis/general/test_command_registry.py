from artemis.general.command_registry import hold_command_registry, add_command_to_registry, NamedCommand, get_registry_or_none


def test_command_registry():
    variable = 3

    def increment_it():
        nonlocal variable
        variable += 1

    with hold_command_registry():
        add_command_to_registry(NamedCommand(
            name='Increment it',
            command=increment_it,
            unique_command_id='command.increment.it'
        ))
        assert variable == 3
        get_registry_or_none()['command.increment.it'].command()
        assert variable == 4


if __name__ == "__main__":
    test_command_registry()
