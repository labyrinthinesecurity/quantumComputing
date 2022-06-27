#!/usr/bin/python3
from qiskit import QuantumCircuit,transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import X, Y, I, Z
from qiskit.synthesis import QDrift
from qiskit.providers.basicaer import QasmSimulatorPy
import math

op = -X + math.sqrt(2)*I + Z
time = math.pi
time = math.pi*2
reps = 5

evo_gate = PauliEvolutionGate(op, time, synthesis=QDrift(reps = reps))

simulator=QasmSimulatorPy()
circ = QuantumCircuit(1,1)
circ.h(0)
circ.append(evo_gate, [0])
circ.measure([0],[0])
print(circ)

compiled_circuit = transpile(circ, simulator)
job = simulator.run(compiled_circuit, shots=1000)
result=job.result()
counts=result.get_counts(circ)
print("\nTotal count for 0 and 1 are:",counts)
