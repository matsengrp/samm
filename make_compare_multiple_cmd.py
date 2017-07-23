out_folder = "_output"
seedrange = range(10)

fitted_str = ""
true_models = []
samm_ones = []
samm_mults = []
for seed in seedrange:
    true_model = "multiple_mutations/%s/0%d/true_model.pkl" % (out_folder, seed)
    samm_one = "multiple_mutations/%s/0%d/False/fitted.pkl" % (out_folder, seed)
    samm_mult = "multiple_mutations/%s/0%d/True/fitted.pkl" % (out_folder, seed)
    true_models.append(true_model)
    samm_ones.append(samm_one)
    samm_mults.append(samm_mult)

true_models = ",".join(true_models)
samm_mults = ",".join(samm_mults)
samm_ones = ",".join(samm_ones)
print "python compare_multiple.py --samm-one %s --samm-mult %s --true %s --agg-motif-len 3 --agg-pos 1" % (samm_ones, samm_mults, true_models)
