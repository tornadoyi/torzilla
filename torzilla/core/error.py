
# Base
class TorzillaError(Exception):
    def __init__(self, err, desc=None):
        text = err
        if desc is not None: text += f' ({desc})'
        text += '.'
        super().__init__(text)

# Argumemt
class ArgumentTypeError(TorzillaError):
    def __init__(self, name, arg, expect_types, **kwargs):
        expects = ','.join([tp.__name__ for tp in expect_types])
        super().__init__(f'Type of {name} is {type(arg)}, expect: {expects}', **kwargs)

class InvalidArgumentError(TorzillaError):
    def __init__(self, name, value, **kwargs):
        super().__init__(f'Argument {name}={value} is invalid', **kwargs)

class NotExistError(TorzillaError):
    def __init__(self, name, **kwargs):
        super().__init__(f'Value {name} is not existed', **kwargs)


class ArgumentMissError(TorzillaError):
    def __init__(self, name, **kwargs):
        super().__init__(f'Miss argument {name}', **kwargs)

class NotSubclassError(TorzillaError):
    def __init__(self, name, child, parent, **kwargs):
        super().__init__(f'Type {child} ({name}) is not subclass of {parent}', **kwargs)


# Import
class ImportError(TorzillaError):
    def __init__(self, name, **kwargs):
        super().__init__(f'Can not import module {name}', **kwargs)


