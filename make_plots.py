model_types = ["3_targetFalse", "3_targetTrue", "2_3_targetFalse"]
effect_sizes = [100] # [50,100,200]
sparsities = [50]# [25, 50, 100]
num_samples = [40,120,360] #[40, 120, 360]
xlabels = num_samples
xlab = "Samples"
nseeds = 10
fitted_str = ""
for i, model_type in enumerate(model_types):
    for eff_size in effect_sizes:
        for sparsity in sparsities:
            for nsamples in num_samples:
                for seed in range(nseeds):
                    fitted_str += "simulation_section/_output/%s/sparsity%d/effect_size_%d/samples%d/0%d/samm/fitted.pkl" % (model_type, sparsity, eff_size, nsamples, seed)
                    if seed < nseeds - 1:
                        fitted_str += ","
                    else:
                        fitted_str += ":"
fitted_str = fitted_str[:-1]                    

true_model_arg = ""
model_types_str = ""
first_time = True
for i, model_type in enumerate(model_types):
    for eff_size in effect_sizes:
        for sparsity in sparsities:
            for nsamples in num_samples:
                if not first_time:
                    true_model_arg += ":"
                    model_types_str += ":"
                first_time = False
                true_model = "simulation_section/_output/%s/sparsity%d/effect_size_%d/true_model.pkl" % (model_type, sparsity, eff_size)
                true_model_arg += true_model
                model_types_str += model_type

print """
python plot_simulation_section.py --true %s --fitted %s --model-types %s --x-lab '%s' --x-labels %s
""" % (true_model_arg, fitted_str, model_types_str, xlab, ":".join([str(x) for x in xlabels]))
