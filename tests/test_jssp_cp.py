from models.jssp_cp import JSSPCpModel


def _print_solution(model: JSSPCpModel, data_name: str):
    res = model.optimize(time_limit_seconds=10)
    print(f"\nInstance: {data_name} -> status: {res['status']}, obj: {res['obj_val']}")
    if res['obj_val'] is None:
        return

    sol = res.get("solution", {})
    selected = sol.get("selected_modes", [])
    print(" Selected modes:")
    for (t, m, w) in selected:
        print(f"  - Task {t}: machine={m}, worker={w}")

    tasks = sol.get("tasks", {})
    print(" Task times:")
    for t, info in tasks.items():
        print(f"  - {t}: start={info['start']} end={info['end']}")


# New instances aligned with MILP tests: provide M_t and W_m and only feasible triples in p

def instance_1():
    machines = ["m0", "m1"]
    workers = ["w0"]

    job_tasks = {
        "j0": ["t0_0", "t0_1"],
        "j1": ["t1_0", "t1_1"],
    }
    tasks = sum(job_tasks.values(), [])
    P = [("t0_0", "t0_1"), ("t1_0", "t1_1")]

    M_t = {
        "t0_0": ["m0"],
        "t0_1": ["m1"],
        "t1_0": ["m0"],
        "t1_1": ["m1"],
    }

    W_m = {
        "w0": ["m0", "m1"],
    }

    p = {}
    for t in tasks:
        for m in M_t[t]:
            for w in workers:
                if m in W_m[w]:
                    p[(t, m, w)] = 1 if m == "m0" else 2

    data = {
        "tasks": tasks,
        "job_tasks": job_tasks,
        "machines": machines,
        "workers": workers,
        "P": P,
        "M_t": M_t,
        "W_m": W_m,
        "p": p,
        "V": 1000,
    }
    return data


def instance_2():
    machines = ["m0", "m1", "m2"]
    workers = ["w0", "w1"]

    job_tasks = {
        "j0": ["a0", "a1"],
        "j1": ["b0", "b1"],
        "j2": ["c0", "c1"],
    }
    tasks = sum(job_tasks.values(), [])
    P = [("a0", "a1"), ("b0", "b1"), ("c0", "c1")]
    L = {("b0", "a1"): 1.0}

    M_t = {}
    for idx, t in enumerate(tasks):
        if idx % 3 == 0:
            M_t[t] = ["m0", "m1"]
        elif idx % 3 == 1:
            M_t[t] = ["m1", "m2"]
        else:
            M_t[t] = ["m0", "m2"]

    W_m = {
        "w0": ["m0", "m1"],
        "w1": ["m1", "m2"],
    }

    p = {}
    for t in tasks:
        for m in M_t[t]:
            for w in workers:
                if m in W_m[w]:
                    p[(t, m, w)] = 1

    s = {}
    for i in tasks:
        for k in tasks:
            if i == k:
                continue
            s[(i, k, "m0")] = 1.0

    data = {
        "tasks": tasks,
        "job_tasks": job_tasks,
        "machines": machines,
        "workers": workers,
        "P": P,
        "L": L,
        "M_t": M_t,
        "W_m": W_m,
        "p": p,
        "s": s,
        "V": 1000,
    }
    return data


def instance_3():
    machines = ["m0", "m1", "m2", "m3"]
    workers = ["w0", "w1"]

    job_tasks = {
        "j0": ["t00"],
        "j1": ["t10"],
        "j2": ["t20"],
        "j3": ["t30"],
    }
    tasks = sum(job_tasks.values(), [])
    P = []

    M_t = {
        "t00": ["m0", "m2"],
        "t10": ["m1"],
        "t20": ["m2", "m3"],
        "t30": ["m0", "m3"],
    }

    W_m = {
        "w0": ["m0", "m2"],
        "w1": ["m1", "m3"],
    }

    p = {}
    for idx, t in enumerate(tasks):
        for m in M_t[t]:
            for w in workers:
                if m in W_m[w]:
                    p[(t, m, w)] = 1

    release_dates = {"j2": 2.0}
    deadlines = {"j3": 5.0}

    data = {
        "tasks": tasks,
        "job_tasks": job_tasks,
        "machines": machines,
        "workers": workers,
        "P": P,
        "M_t": M_t,
        "W_m": W_m,
        "p": p,
        "V": 1000,
        "release_dates": release_dates,
        "deadlines": deadlines,
    }
    return data


def main():
    instances = [("easy-1", instance_1()), ("easy-2", instance_2()), ("easy-3", instance_3())]
    for name, data in instances:
        print(f"\n--- Solving CP instance {name} ---")
        try:
            model = JSSPCpModel(data)
        except Exception as e:
            print(f"Failed to build CP model for {name}: {e}")
            continue
        _print_solution(model, name)


if __name__ == '__main__':
    main()
