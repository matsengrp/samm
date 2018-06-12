sim_type = "final_revisions_shmulate_m3-5_s2000"
seedrange =  ["%02d" % s for s in range(50)]

true_model = "simulated_shazam_vs_samm/_output/%s/%%s/True/true_model.pkl" % (sim_type,)
samm_m = "simulated_shazam_vs_samm/_output/%s/%%s/True/fitted.pkl" % (sim_type,)
logistic_m = "simulated_shazam_vs_samm/_output/%s/%%s/True/logistic_model.pkl" % (sim_type,)
shazam_sub = "simulated_shazam_vs_samm/_output/%s/%%s/True/fitted_shazam_sub.csv" % (sim_type,)
shazam_mut = "simulated_shazam_vs_samm/_output/%s/%%s/True/fitted_shazam_mut.csv" % (sim_type,)

seedrange = ",".join(seedrange)
print "python compare_simulated_shazam_vs_samm.py --in-shazam-mut %s --in-shazam-sub %s --in-samm %s --in-logistic %s --true %s --seeds %s --agg-motif-len 5 --agg-pos 2 --wide" % (shazam_mut, shazam_sub, samm_m, logistic_m, true_model, seedrange)
