model_type = "3_targetFalse"
fitted_str = ""
for nsamples in [40,120,360]:
    for seed in range(10):
        fitted_str += "simulation_section/_output/%s/sparsity50/effect_size_100/samples%d/0%d/samm/fitted.pkl" % (model_type, nsamples, seed)
        if seed < 9:
            fitted_str += ","
        elif nsamples != 360:
            fitted_str += ":"

true_model = "simulation_section/_output/%s/sparsity50/effect_size_100/true_model.pkl" % model_type
true_model_arg = "%s:%s:%s" % (true_model, true_model, true_model)
print """
python plot_simulation_section.py --true %s --fitted %s --title %s
""" % (true_model_arg, fitted_str, model_type)
