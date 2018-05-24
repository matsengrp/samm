sim_type = "final_revisions_survival_m3-5_s2000"
seedrange = ["0%d" % i for i in range(10)] + [str(i) for i in range(10,27)]

fitted_str = ""
true_models = []
shazam_muts = []
shazam_subs = []
samms = []
logistics = []
for seed in seedrange:
    true_model = "simulated_shazam_vs_samm/_output/%s/%s/True/true_model.pkl" % (sim_type, seed)
    samm_m = "simulated_shazam_vs_samm/_output/%s/%s/True/fitted.pkl" % (sim_type, seed)
    logistic_m = "simulated_shazam_vs_samm/_output/%s/%s/True/logistic_model.pkl" % (sim_type, seed)
    shazam_sub = "simulated_shazam_vs_samm/_output/%s/%s/True/fitted_shazam_sub.csv" % (sim_type, seed)
    shazam_mut = "simulated_shazam_vs_samm/_output/%s/%s/True/fitted_shazam_mut.csv" % (sim_type, seed)
    true_models.append(true_model)
    samms.append(samm_m)
    logistics.append(logistic_m)
    shazam_subs.append(shazam_sub)
    shazam_muts.append(shazam_mut)

true_models = ",".join(true_models)
samms = ",".join(samms)
logistics = ",".join(logistics)
shazam_muts = ",".join(shazam_muts)
shazam_subs = ",".join(shazam_subs)
print "python compare_simulated_shazam_vs_samm.py --in-shazam-mut %s --in-shazam-sub %s --in-samm %s --in-logistic %s --true %s --agg-motif-len 5 --agg-pos 2 --wide" % (shazam_muts, shazam_subs, samms, logistics, true_models)
