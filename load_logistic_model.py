import pickle

from read_data import load_logistic_model
from fit_logistic_model import LogisticModel

input_model = "get_hedgehog/_output/00/motif-3-5-flank-1--2/True/logistic_model.pkl"
output_model = "simulation_tyler/data/logistic_model.pkl"

fitted_model = load_logistic_model(input_model)
print(fitted_model.agg_refit_theta)

# Dump a pickle file of simulation parameters
with open(output_model, 'w') as f:
    pickle.dump((fitted_model.agg_refit_theta, None), f)
