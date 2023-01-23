class StateError(RuntimeError):
    """Indicates that an object is not in the right state to perform an operation."""
    def __init__(self, msg: str = "") -> None:
        super().__init__(msg)
