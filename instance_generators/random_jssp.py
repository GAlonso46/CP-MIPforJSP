"""
Generador aleatorio de instancias JSSP básicas.
Cada job tiene m operaciones (una por máquina). Para cada job se elige una permutación aleatoria
de las máquinas y para cada operación se asigna un tiempo de procesamiento entero uniformemente
muestreado entre 1 y 99 (incluidos), usando el módulo random de Python.

Funciones principales:
- generate_instance(j, m, seed=None): devuelve la instancia como un diccionario.
- instance_to_string(instance): devuelve una representación JSON legible de la instancia.

Formato de la instancia devuelta:
{
  "num_jobs": j,
  "num_machines": m,
  "jobs": [
    {
      "job_id": 0,
      "operations": [
        {"operation_id": 0, "machine": 2, "processing_time": 45},
        ...
      ]
    },
    ...
  ]
}

Este archivo también puede ejecutarse como script para generar una instancia desde la línea de comandos.
"""

from typing import Dict, Optional
import random
import json


def generate_instance(j: int, m: int, seed: Optional[int] = None) -> Dict:
    """Genera y devuelve una instancia JSSP aleatoria.

    Args:
        j: número de jobs (debe ser > 0)
        m: número de máquinas (debe ser > 0)
        seed: semilla opcional para reproducibilidad

    Returns:
        Diccionario con la instancia en el formato descrito arriba.
    """
    if j <= 0 or m <= 0:
        raise ValueError("j y m deben ser enteros positivos")

    if seed is not None:
        random.seed(seed)

    instance = {"num_jobs": j, "num_machines": m, "jobs": []}
    machines = list(range(m))

    for job_id in range(j):
        perm = machines[:]  # copia
        random.shuffle(perm)
        ops = []
        for op_idx in range(m):
            proc_time = random.randint(1, 99)  # uniforme 1..99
            ops.append({
                "operation_id": op_idx,
                "machine": perm[op_idx],
                "processing_time": proc_time,
            })
        instance["jobs"].append({"job_id": job_id, "operations": ops})

    return instance


def instance_to_string(instance: Dict) -> str:
    """Devuelve una representación JSON legible de la instancia."""
    return json.dumps(instance, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generador aleatorio JSSP básico")
    parser.add_argument("j", type=int, help="Número de jobs")
    parser.add_argument("m", type=int, help="Número de máquinas")
    parser.add_argument("--seed", type=int, default=None, help="Semilla aleatoria (opcional)")
    args = parser.parse_args()

    inst = generate_instance(args.j, args.m, seed=args.seed)
    print(instance_to_string(inst))
