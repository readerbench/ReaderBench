
class WordActivation():

    def __init__(self, activation_value: float, active: bool):
        self.activation_value = activation_value
        self.active = active

    def __repr__(self):
        return "WordActivation(%r, active: %r)" % (self.activation_value, self.active)


    def __str__(self):
        return "WordActivation(%r, active: %r)" % (self.activation_value, self.active)