"""
Test simple para el generador de instancias aleatorias JSSP.
Genera una instancia con 2 jobs y 3 máquinas, imprime la representación y realiza
una comprobación básica de formato.

Ejecutar: python3 tests/test_generator.py
"""
from instance_generators.random_jssp import generate_instance, instance_to_string


def main():
    inst = generate_instance(2, 3, seed=42)
    print("Instancia generada:")
    print(instance_to_string(inst))

    # comprobaciones simples
    assert inst["num_jobs"] == 2
    assert inst["num_machines"] == 3
    assert len(inst["jobs"]) == 2
    for job in inst["jobs"]:
        ops = job["operations"]
        assert len(ops) == 3
        machines = [op["machine"] for op in ops]
        # debe ser una permutación de [0,1,2]
        assert sorted(machines) == [0, 1, 2]
        for op in ops:
            pt = op["processing_time"]
            assert 1 <= pt <= 99

    print("Prueba completada correctamente")


if __name__ == "__main__":
    main()
