from model.math.activations import get_activations
from model.math.optimizer import get_optimizers

act_names, _, _ = get_activations()
opt_names, _ = get_optimizers()
print("available activations: ", act_names)
print("available optimizers: ", opt_names)

