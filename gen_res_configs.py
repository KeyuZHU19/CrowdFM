base = open("config/res_smoke.yaml").read()
base = (base.replace("epochs: 12", "epochs: 10000")
            .replace("save_interval: 12", "save_interval: 10000")
            .replace("gradient_accumulation_steps: 4", "gradient_accumulation_steps: 16")
            .replace("num_workers: 0", "num_workers: 4")
            .replace("num_worker_range: [20, 60], num_task_range: [50, 100], num_option_range: [2, 6], num_answer_each_task_range: [3, 6]",
                     "num_worker_range: [20, 100], num_task_range: [100, 300], num_option_range: [2, 20], num_answer_each_task_range: [3, 10]"))
allf = "mechanism_families: [irt, class_bias, coalition, assignment_bias, mixed]"
ho = "mechanism_families: [irt, class_bias]"

def variant(mode, use_mech, setting, fams, seed):
    kind = "meta" if mode == "meta" else ("mech" if use_mech else "nomech")
    name = "res_" + kind + "_" + setting + "_s" + str(seed)
    c = base.replace("mode: residual", "mode: " + mode)
    c = c.replace("use_mechanism: true", "use_mechanism: " + ("true" if use_mech else "false"))
    c = c.replace(allf, fams)
    c = c.replace("output_dir: log/res_smoke", "output_dir: runs/" + name)
    c = c + "\nseed: " + str(seed) + "\n"
    open("config/" + name + ".yaml", "w").write(c)
    return name

names = []
for seed in [42]:
    for setting, fams in [("ip", allf), ("ho", ho)]:
        names.append(variant("residual", True, setting, fams, seed))
        names.append(variant("residual", False, setting, fams, seed))
        names.append(variant("meta", True, setting, fams, seed))
print("wrote", names)
