from qiskit.circuit import QuantumCircuit
from qiskit import Aer, execute
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import I, Z, Y, X
from qiskit.quantum_info import SparsePauliOp
import sys
import numpy as np

H=np.array(
[
[(1.009647315010322+0.19651249845223961j),(-0.5404786773379169-0.6974604975046392j),(-0.4308597911850717+0.19805067348973598j),(-0.43085979118507167+0.1980506734897361j),0j,0j,0j,0j],
[(-0.5404786773379152-0.6974604975046378j),(1.8537646222071038+0.47139183396486983j),(-0.5479160689842743+0.3311284968093143j),(-0.5479160689842743+0.33112849680931417j),0j,0j,0j,0j],
[(-0.43085979118507106+0.19805067348973593j),(-0.5479160689842743+0.33112849680931533j),(1.0682940313912872-0.3339521662085547j),(0.06829403139128695-0.3339521662085547j),0j,0j,0j,0j],
[(-0.4308597911850711+0.19805067348973598j),(-0.5479160689842741+0.3311284968093152j),(0.06829403139128706-0.3339521662085547j),(1.068294031391287-0.3339521662085547j),0j,0j,0j,0j],
[0j,0j,0j,0j,(0.9999999999999999+0j),(-0.9268857942017941+0j),(-0.3745514107232036+0j),(0.024371401932735202+0j)],
[0j,0j,0j,0j,(-0.9268857942017952+0j),(2.1391008288104074+0j),(-0.33803803882237127+0j),(0.09510974680227287+0j)],
[0j,0j,0j,0j,(-0.37455141072320264+0j),(-0.33803803882236966+0j),(2.812948320967011+0j),(-0.3623802867562586+0j)],
[0j,0j,0j,0j,(0.024371401932734956+0j),(0.095109746802273+0j),(-0.362380286756259+0j),(0.047950850222581026+0j)]
])

operator=SparsePauliOp.from_operator(H)
print(operator)

backend = Aer.get_backend('qasm_simulator')
duration = 40.0
dt =0.01
steps = int(duration / dt)
shots=2000

pos=['000','001','010','011','100','101','110','111']

for i in range(steps):
    tt=(float(i)*dt)
    evo = PauliEvolutionGate(operator, time=tt)
    circuit = QuantumCircuit(3, 3)
    circuit.h(0)
    circuit.h(1)
    circuit.h(2)
    circuit.append(evo, range(3))
    circuit.measure(range(3), range(3))
    job = execute(circuit, backend, shots=shots)
    counts = job.result().get_counts()
    j=-1
    print(tt,end=';')
    for c in pos:
      j=j+1
      if c in counts:
        print(float(counts[c])/shots,end=';')
      else:
        print(0.0,end=';')
    print()
