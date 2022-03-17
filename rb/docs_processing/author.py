
class Author:

    def __init__(self, name: str):
        self.name = name
        self.articles = set()

    def __eq__(self, other):
        if isinstance(other, Author):
            return self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash(tuple(self.name))
    
    def __str__(self):
        return self.name
    
    def __repr__(self) -> str:
        return self.name