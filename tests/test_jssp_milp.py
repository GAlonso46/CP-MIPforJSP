from models.jssp_milp import JSSPMilpModel


def _print_solution(model: JSSPMilpModel, data_name: str):
    res = model.optimize(time_limit=10)
    print(f"\nInstance: {data_name} -> status: {res['status']}, obj: {res['obj_val']}")
    if res['obj_val'] is None:
        return
    # print task schedule and assignment
    for t in model.tasks:
        Sval = model.S[t].X if model.S[t].X is not None else None
        Cval = model.C[t].X if model.C[t].X is not None else None
        assign = [(m, w) for (tt, m, w), var in model.x.items() if tt == t and getattr(var, 'X', 0) > 0.5]
        assign_str = f"machine={assign[0][0]}, worker={assign[0][1]}" if assign else "unassigned"
        print(f" Task {t}: S={Sval:.1f} C={Cval:.1f} -> {assign_str}")


def instance_1():
    # 2 jobs, 2 machines, 1 worker; each job 2 tasks in sequence
    machines = ["m0", "m1"]
    workers = ["w0"]

    jobs = ["j0", "j1"]
    job_tasks = {
        "j0": ["t0_0", "t0_1"],
        "j1": ["t1_0", "t1_1"],
    }
    tasks = sum(job_tasks.values(), [])

    P = [("t0_0", "t0_1"), ("t1_0", "t1_1")]

    M_t = {t: machines[:] for t in tasks}
    W_t = {t: workers[:] for t in tasks}

    # processing times: (t,m,w) -> small integers
    p = {}
    for t in tasks:
        for m in machines:
            p[(t, m, "w0")] = 1 if m == "m0" else 2

    data = {
        "tasks": tasks,
        "job_tasks": job_tasks,
        "machines": machines,
        "workers": workers,
        "P": P,
        "M_t": M_t,
        "W_t": W_t,
        "p": p,
    }
    return data


def instance_2():
    # 3 jobs, 3 machines, 2 workers; include a time lag and simple SDST on m0
    machines = ["m0", "m1", "m2"]
    workers = ["w0", "w1"]

    job_tasks = {
        "j0": ["a0", "a1"],
        "j1": ["b0", "b1"],
        "j2": ["c0", "c1"],
    }
    tasks = sum(job_tasks.values(), [])
    # precedence within jobs
    P = [("a0", "a1"), ("b0", "b1"), ("c0", "c1")]
    # inter-job lag: a1 must start at least 1 after b0 completion
    L = {("b0", "a1"): 1}

    M_t = {t: machines[:] for t in tasks}
    W_t = {t: workers[:] for t in tasks}

    p = {}
    for t in tasks:
        for m in machines:
            for w in workers:
                p[(t, m, w)] = 1

    # simple SDST: on m0, switching from any task i to k adds 1 time unit
    s = {}
    for i in tasks:
        for k in tasks:
            if i == k:
                continue
            s[(i, k, "m0")] = 1

    data = {
        "tasks": tasks,
        "job_tasks": job_tasks,
        "machines": machines,
        "workers": workers,
        "P": P,
        "L": L,
        "M_t": M_t,
        "W_t": W_t,
        "p": p,
        "s": s,
    }
    return data


def instance_3():
    # 4 jobs, up to 4 machines, 2 workers; include release dates and deadlines
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

    M_t = {t: machines[:] for t in tasks}
    W_t = {t: workers[:] for t in tasks}

    p = {}
    # slightly different processing times to create contention
    for idx, t in enumerate(tasks):
        for m in machines:
            for w in workers:
                p[(t, m, w)] = 1

    release_dates = {"j2": 2}  # job j2 cannot start before 2
    deadlines = {"j3": 5}     # job j3 must finish early

    data = {
        "tasks": tasks,
        "job_tasks": job_tasks,
        "machines": machines,
        "workers": workers,
        "P": P,
        "M_t": M_t,
        "W_t": W_t,
        "p": p,
        "release_dates": release_dates,
        "deadlines": deadlines,
    }
    return data


def main():
    instances = [("easy-1", instance_1()), ("easy-2", instance_2()), ("easy-3", instance_3())]
    for name, data in instances:
        print(f"\n--- Solving instance {name} ---")
        try:
            model = JSSPMilpModel(data)
        except Exception as e:
            print(f"Failed to build model for {name}: {e}")
            continue
        _print_solution(model, name)


if __name__ == '__main__':
    main()