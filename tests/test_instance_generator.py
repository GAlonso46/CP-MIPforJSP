"""
Simple test for the JSSP random instance generator.
Generates an instance with 2 jobs and 3 machines, prints the representation and performs
a basic format check.

Run: python3 tests/test_generator.py
"""
from instance_generators.random_jssp import generate_instance, instance_to_string


def main():
    inst = generate_instance(2, 3, seed=42)
    print("Generated instance:")
    print(instance_to_string(inst))

    # simple checks
    assert inst["num_jobs"] == 2
    assert inst["num_machines"] == 3
    assert len(inst["jobs"]) == 2
    for job in inst["jobs"]:
        ops = job["operations"]
        assert len(ops) == 3
        machines = [op["machine"] for op in ops]
        # must be a permutation of [0,1,2]
        assert sorted(machines) == [0, 1, 2]
        for op in ops:
            pt = op["processing_time"]
            assert 1 <= pt <= 99

    print("Test completed successfully")


if __name__ == "__main__":
    main()