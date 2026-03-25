"""汎用レジストリファクトリ."""


def make_registry():
    """名前→クラスのレジストリとデコレータを返す."""
    registry: dict[str, type] = {}

    def register(name: str):
        def wrapper(cls):
            registry[name] = cls
            return cls
        return wrapper

    return registry, register
