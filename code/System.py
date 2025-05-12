## test
class System:
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version

    def __repr__(self):
        return f"System(name={self.name}, version={self.version})"