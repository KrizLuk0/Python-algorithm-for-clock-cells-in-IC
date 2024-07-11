# Transistor Width and Length Optimization in CMOS Technology


This algorithm sets the correct values for the widths and lengths of transistors in CMOS technology. It utilizes inverter chains to find real edges, works with netlists, adjusts the output capacitance of the circuit, and modifies the widths and lengths of the transistors. The entire process is written in Python and allows for balancing, with the balance determined by the similarity of output edge lengths and delays between edges. The goal is to minimize the difference between these two parameters. Differential evolution is used to adjust the lengths and widths. Upon completion, a graph is generated to show the results of each generation.

## Features

- **Inverter Chains**: Utilizes chains of inverters to find real edges.
- **Netlist Processing**: Works with netlists to adjust output capacitance.
- **Transistor Modification**: Adjusts the widths and lengths of transistors.
- **Balancing**: Ensures the balance by matching output edge lengths and delays.
- **Differential Evolution**: Uses differential evolution for optimization.
- **Graphical Output**: Generates a graph showing the results across generations.

##  Output
After the algorithm completes, a graph is generated showing the results of each generation. This graph provides insight into the progress and effectiveness of the optimization process.

# Verry important: You must have instalitation PySpice!!!


