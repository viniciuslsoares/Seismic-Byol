from torch import nn


def build(*layer_sizes):
    """Builds a Multilayer perceptron with the informed layer sizes.
    
    Parameters
    ----------
    *layer_sizes : int
        Size of nth layer.
    """
    assert len(layer_sizes) >= 2, "Multilayer perceptron must have at least 2 layers"
    assert all(ls > 0 and isinstance(ls, int) for ls in layer_sizes), "All layer sizes must be a positive integer"

    layers = []
    for i in range(len(layer_sizes) - 2):
        layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]), nn.ReLU()]
    layers += [nn.Linear(layer_sizes[-2], layer_sizes[-1])]
    return nn.Sequential(*layers)

class MLP(nn.Sequential):

    def __init__(self, *layer_sizes):
        assert len(layer_sizes) >= 2, "Multilayer perceptron must have at least 2 layers"
        assert all(ls > 0 and isinstance(ls, int) for ls in layer_sizes), "All layer sizes must be a positive integer"

        layers = []
        for i in range(len(layer_sizes) - 2):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]), nn.ReLU()]
        layers += [nn.Linear(layer_sizes[-2], layer_sizes[-1])]

        super().__init__(*layers)
