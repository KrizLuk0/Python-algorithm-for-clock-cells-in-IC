import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
import matplotlib.pyplot as plt
import gc

ngspice = NgSpiceShared.new_instance() # Implement functions to clear memory

# Function to simulate a chain of inverters and measure rise/fall times
def SimulationChainOfInverters(ChainNetlist, LowThreshold, HighThreshold, SimParams):
    circuit = Circuit('Chain of Inverter')  # Create a new circuit
    circuit.include(ChainNetlist)  # Include the netlist file for the inverter chain

    # Setup and run a transient simulation
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(**SimParams)

    # Extract output voltage and time from the simulation
    OutVoltage = analysis['Vout']
    Time = np.array(analysis.time)
    Voltages = np.array(OutVoltage)

    # Measure rise and fall times using edges detection
    RiseTimeChain, FallTimeChain = MeasEdges(Time, Voltages, LowThreshold, HighThreshold)

    # Clear memory
    ngspice.destroy()
    ngspice.remove_circuit()
    del circuit, simulator, OutVoltage, Time, Voltages, analysis
    gc.collect()

    return RiseTimeChain, FallTimeChain

# Function to detect rising and falling edges based on voltage thresholds
def MeasEdges(Time, Voltages, LowThreshold, HighThreshold):
    RiseEdge = FallEdge = None

    # Loop through voltage data to find where it crosses the low threshold upwards
    for i in range(1, len(Voltages)):
        if Voltages[i - 1] < LowThreshold <= Voltages[i]:
            StartRise = Time[i]
            for j in range(i, len(Voltages)):
                if Voltages[j - 1] < HighThreshold <= Voltages[j]:
                    RiseEdge = Time[j] - StartRise
                    break
            break

    # Loop through voltage data to find where it crosses the high threshold downwards
    for i in range(1, len(Voltages)):
        if Voltages[i - 1] > HighThreshold >= Voltages[i]:
            StartFall = Time[i]
            for j in range(i, len(Voltages)):
                if Voltages[j - 1] > LowThreshold >= Voltages[j]:
                    FallEdge = Time[j] - StartFall
                    break
            break

    return RiseEdge, FallEdge

# Function to adjust parameters of a generator based on measured rise and fall times
def ChangeParamsOfGen(RiseChain, FallChain, SimulateNetlist):
    RiseTime_ps = int(RiseChain*1e12)
    FallTime_ps = int(FallChain*1e12)

    # Calculate average parameter value for generator
    ParamOfGen = (RiseTime_ps + FallTime_ps) // 2
    UpdatedContent = []
    try:
        with open(SimulateNetlist, 'r') as file:
            lines = file.readlines()

        ParamOfGenString = f"{ParamOfGen}p"

        # Update the pulse parameters in the netlist file
        for line in lines:
            if 'pulse(' in line:
                parts = line.split('pulse(')
                PulseParams = parts[1].strip(')\n').split(' ')
                PulseParams[3] = ParamOfGenString
                PulseParams[4] = ParamOfGenString

                NewPulseParams = 'pulse(' + ' '.join(PulseParams) + ')'
                UpdateLine = parts[0] + NewPulseParams + '\n'
                UpdatedContent.append(UpdateLine)
            else:
                UpdatedContent.append(line)

        # Write the updated content back to the netlist file
        with open(SimulateNetlist, 'w') as file:
            file.writelines(UpdatedContent)
        return ParamOfGen
    except Exception as e:
        print(f"Error updating parameters of generator: {e}")

# Function to simulate the effects of changing the capacitance in the netlist
def SimulationCapacity(SimulateNetlist, SimParams, LowThreshold, HighThreshold):
    circuit = Circuit('Change capacity')
    circuit.include(SimulateNetlist)  # Include the netlist file for simulation

    # Setup and run a transient simulation
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(**SimParams)

    # Extract output voltage and time data
    OutVoltage = analysis['Vout']
    Time = np.array(analysis.time)
    Voltages = np.array(OutVoltage)
    # Measure the rise edge time to compare against target
    RiseEdge, _ = MeasEdges(Time, Voltages, LowThreshold, HighThreshold)
    ngspice.destroy()
    ngspice.remove_circuit()
    del circuit, simulator, OutVoltage, Time, Voltages, analysis
    gc.collect()


    return RiseEdge  # Return the length of the rise edge for comparison


# Function to update the capacitance value in the netlist
def ChangeValueOfCapacity(SimulateNetlist, NewValueOfCapacity):
    UpdatedContent = []

    try:
        with open(SimulateNetlist, 'r') as file:
            lines = file.readlines()

        # Modify the capacitance value in the netlist
        for line in lines:
            if line.startswith('C'):
                parts = line.split()
                parts[3] = f"{float(NewValueOfCapacity):.2f}f"
                UpdateLine = ' '.join(parts) + '\n'
                UpdatedContent.append(UpdateLine)
            else:
                UpdatedContent.append(line)

        # Write the updated content back to the netlist file
        with open(SimulateNetlist, 'w') as file:
            file.writelines(UpdatedContent)
    except Exception as e:
        print(f"Error updating netlist: {e}")

# Function to iteratively adjust capacitance to meet a target rise time
def MeasCapacity(SimulateNetlist, SimParams, LowThreshold, HighThreshold, TargetRiseTime, LowCapacity, HighCapacity,
                 Toleration):
    OptimalCapacity = None
    # Perform a binary search to find the optimal capacitance
    while (HighCapacity - LowCapacity) > Toleration:
        MiddleCapacity = (LowCapacity + HighCapacity) / 2
        ChangeValueOfCapacity(SimulateNetlist, MiddleCapacity)
        CurrentRiseEdge = SimulationCapacity(SimulateNetlist, SimParams, LowThreshold, HighThreshold)

        # Adjust the capacitance range based on the measured rise time
        if CurrentRiseEdge > TargetRiseTime:
            HighCapacity = MiddleCapacity
        else:
            LowCapacity = MiddleCapacity

    # Return the middle value as the optimal capacitance after tolerance is reached
    if OptimalCapacity is None:
        OptimalCapacity = (LowCapacity + HighCapacity) / 2
    return OptimalCapacity

# Function to measure delays between input and output signals
def MeasDelay(Time, InputSignal, OutputSignal, LowThreshold, HighThreshold, Mode):
    Signal50Percent = (LowThreshold + HighThreshold) / 2  # Calculate 50% threshold level
    InputRiseEdge = OutputRiseEdge = InputFallEdge = OutputFallEdge = None

    # Detect rise and fall edges in the input signal
    for i in range(1, len(InputSignal)):
        if InputSignal[i-1] < Signal50Percent <= InputSignal[i] and InputRiseEdge is None:
            InputRiseEdge = Time[i]
        elif InputSignal[i-1] > Signal50Percent >= InputSignal[i] and InputFallEdge is None:
            InputFallEdge = Time[i]
            break

    # Detect rise and fall edges in the output signal based on the mode
    if Mode == "unbalanced":
        for i in range(1, len(OutputSignal)):
            if OutputSignal[i-1] > Signal50Percent >= OutputSignal[i] and OutputFallEdge is None:
                OutputFallEdge = Time[i]
            elif OutputSignal[i-1] < Signal50Percent <= OutputSignal[i] and OutputRiseEdge is None:
                OutputRiseEdge = Time[i]
                break
    elif Mode == "balanced":
        for i in range(1, len(OutputSignal)):
            if OutputSignal[i-1] < Signal50Percent <= OutputSignal[i] and OutputRiseEdge is None:
                OutputRiseEdge = Time[i]
            elif OutputSignal[i-1] > Signal50Percent >= OutputSignal[i] and OutputFallEdge is None:
                OutputFallEdge = Time[i]
                break

    # Calculate rise and fall delays
    if InputRiseEdge is not None and OutputRiseEdge is not None:
        RiseDelay = abs(OutputRiseEdge - InputRiseEdge)
    else:
        RiseDelay = None

    if InputFallEdge is not None and OutputFallEdge is not None:
        FallDelay = abs(OutputFallEdge - InputFallEdge)
    else:
        FallDelay = None

    return RiseDelay, FallDelay


# Function to update transistor parameters in the netlist
def UpdateParamsOfDelay(SimulateNetlist, Width, Length):
    UpdateContent = []
    with open(SimulateNetlist, 'r') as file:
        lines = file.readlines()
    TranzistorIndex = 0
    for line in lines:
        if 'XNMOS_Delay' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part.startswith('W='):
                    parts[i] = f"W={float(Width[TranzistorIndex]):.3f}"
                elif part.startswith('L='):
                    parts[i] = f"L={float(Length[TranzistorIndex]):.3f}"
            UpdateContent.append(' '.join(parts) + '\n')
        else:
            UpdateContent.append(line)

    with open(SimulateNetlist, 'w') as file:
        file.writelines(UpdateContent)


def UpdateParamsOfEdge(SimulateNetlist, Width, Length):
    UpdateContent = []
    with open(SimulateNetlist, 'r') as file:
        lines = file.readlines()
    TranzistorIndex = 0
    for line in lines:
        if 'XNMOS_Edge' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part.startswith('W='):
                    parts[i] = f"W={float(Width[TranzistorIndex]):.3f}"
                elif part.startswith('L='):
                    parts[i] = f"L={float(Length[TranzistorIndex]):.3f}"
            UpdateContent.append(' '.join(parts) + '\n')
        else:
            UpdateContent.append(line)

    with open(SimulateNetlist, 'w') as file:
        file.writelines(UpdateContent)

# Function to simulate transistors and measure delays and edges
def SimulationTranzistors(SimulateNetlist, SimParams, LowThreshold, HighThreshold, Mode):
    circuit = Circuit('Simulation transistors')
    circuit.include(SimulateNetlist)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(**SimParams)

    OutVoltage = analysis['Vout']
    Time = np.array(analysis.time)
    Voltages = np.array(OutVoltage)

    Vin = analysis['Va']
    InputSignal = np.array(Vin)

    # Measure delays and edges
    RiseDelay, FallDelay = MeasDelay(Time, InputSignal, Voltages, LowThreshold, HighThreshold, Mode)
    RiseEdge, FallEdge = MeasEdges(Time, Voltages, LowThreshold, HighThreshold)

    ngspice.destroy()
    ngspice.remove_circuit()
    del circuit, simulator, OutVoltage, Time, Voltages, analysis
    gc.collect()

    return RiseDelay, FallDelay, RiseEdge, FallEdge

# Function to compute a fitness score for a transistor configuration
def FitnessScoreSimulation(SimulateNetlist, NewValueOfWidthDelay, NewValueOfLengthDelay,NewValueOfWidthEdge,NewValueOfLengthEdge, SimParams, LowThreshold, HighThreshold, Mode):
    UpdateParamsOfDelay(SimulateNetlist, NewValueOfWidthDelay, NewValueOfLengthDelay)
    UpdateParamsOfEdge(SimulateNetlist, NewValueOfWidthEdge, NewValueOfLengthEdge)

    RiseDelay, FallDelay, RiseEdge, FallEdge = SimulationTranzistors(SimulateNetlist, SimParams, LowThreshold, HighThreshold, Mode)
    Edge = abs(RiseEdge - FallEdge)

    if FallDelay < 0: # Because unbalanced cells has negative fall delay
        Delay = abs(RiseDelay + FallDelay)
    else:
        Delay = abs(RiseDelay - FallDelay)

    Fitness = abs(Delay + Edge)
    return Fitness, Delay, Edge

# Function to count the number of transistors in a netlist
def CountTranzistorsDelay(SimulateNetlist):
    Count = 0
    try:
        with open(SimulateNetlist, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if 'XNMOS_Delay' in line:
                Count += 1
    except Exception as e:
        print(f"Error: {e}")
    return Count

def CountTranzistorsEdge(SimulateNetlist):
    Count = 0
    try:
        with open(SimulateNetlist, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if 'XNMOS_Edge' in line:
                Count += 1
    except Exception as e:
        print(f"Error: {e}")
    return Count



def AlgorithmDE(SimulateNetlist, SimParams, LowThreshold, HighThreshold, PopSize,
                MaxGenerations, F, CR, WidthMin, WidthMax, LengthBase, LengthIncrease,
                AccuracyThreshold, Thresh, Mode):
    TranzistorsDelays = CountTranzistorsDelay(SimulateNetlist)
    TranzistorsEdges = CountTranzistorsEdge(SimulateNetlist)

    # Initialize populations of widths and lengths
    WidthPopulationDelay = np.random.uniform(WidthMin, WidthMax, (PopSize,TranzistorsDelays))
    WidthPopulationEdge = np.random.uniform(WidthMin, WidthMax, (PopSize,TranzistorsEdges))

    LengthPopulationDelay = LengthBase * np.ones((PopSize,TranzistorsDelays))
    LengthPopulationEdge = LengthBase * np.ones((PopSize,TranzistorsEdges))

    WidtConstDelay = WidthMin * np.ones((PopSize, TranzistorsDelays))
    WidthConstEdge = WidthMin * np.ones((PopSize, TranzistorsEdges))


    FitnessHistory = []

    NoChange = 0
    # Iterate over generations
    for Generation in range(MaxGenerations):
        FitnessScore = np.zeros(PopSize)
        Delay = np.zeros(PopSize)
        Edge = np.zeros(PopSize)
        # Evaluate the fitness of each individual in the population
        for i in range(PopSize):
            FitnessScore[i], Delay[i], Edge[i] = FitnessScoreSimulation(SimulateNetlist, WidthPopulationDelay[i], LengthPopulationDelay[i],
                                                     WidthPopulationEdge[i], LengthPopulationEdge[i], SimParams,
                                                     LowThreshold, HighThreshold, Mode)

        # Record the best fitness score of the generation
        MinFitness = np.min(FitnessScore)
        FitnessHistory.append(MinFitness)

        # Compare a fitness score (local condition)
        if len(FitnessHistory) > 1 and MinFitness == FitnessHistory[-2]:
            NoChange += 1
        else:
            NoChange = 0

        if NoChange >= 20:
            break



        # Perform mutation, crossover, and selection operations
        for i in range(PopSize):
            IdxsDelay = [idxD for idxD in range(PopSize) if idxD != i]
            DelayA, DelayB, DelayC = np.random.choice(IdxsDelay, 3, replace=False)

            IdxsEdge = [idxE for idxE in range(PopSize) if idxE != i]
            EdgeA, EdgeB, EdgeC = np.random.choice(IdxsEdge, 3, replace=False)

            # Generate trial individual
            MutationWidthDelay = np.clip(WidthPopulationDelay[DelayA] + F * (WidthPopulationDelay[DelayB] - WidthPopulationDelay[DelayC]), WidthMin, WidthMax)
            CrossingDelaz = np.random.rand(TranzistorsDelays) < CR
            TrialWidthDelay = np.where(CrossingDelaz, MutationWidthDelay, WidthPopulationDelay[i])
            TrialLengthDelay = np.copy(LengthPopulationDelay[i])

            # Check if any length has changed
            if all(LengthPopulationDelay[i] != LengthBase):
                TrialWidthDelay = WidtConstDelay[i]


            MutationWidthEdge = np.clip(WidthPopulationEdge[EdgeA] + F * (WidthPopulationEdge[EdgeB] - WidthPopulationEdge[EdgeC]), WidthMin, WidthMax)
            CrossingEdge = np.random.rand(TranzistorsEdges) < CR
            TrialWidthEdge = np.where(CrossingEdge, MutationWidthEdge, WidthPopulationEdge[i])
            TrialLengthEdge = np.copy(LengthPopulationEdge[i])

            # Check if any length has changed
            if all(LengthPopulationEdge[i] != LengthBase):
                TrialWidthEdge = WidthConstEdge[i]

            # Increment lenght of tranzistor
            for j in range(TranzistorsDelays):
                if TrialWidthDelay[j] == WidthMin and Delay[i] > Thresh:
                    TrialLengthDelay[j] += LengthIncrease

            for j in range(TranzistorsEdges):
                if TrialWidthEdge[j] == WidthMin and Edge[i] > Thresh:
                    TrialLengthEdge[j] += LengthIncrease

            # Evaluate the trial individual
            TrialFitnessScore,_,_ = FitnessScoreSimulation(SimulateNetlist, TrialWidthDelay, TrialLengthDelay,
                                TrialWidthEdge, TrialLengthEdge, SimParams, LowThreshold,
                                                HighThreshold, Mode)


            # Selection step
            if TrialFitnessScore < FitnessScore[i]:
                WidthPopulationDelay[i], LengthPopulationDelay[i] = TrialWidthDelay, TrialLengthDelay
                WidthPopulationEdge[i], LengthPopulationEdge[i] = TrialWidthEdge, TrialLengthEdge
                FitnessScore[i] = TrialFitnessScore
                UpdateParamsOfEdge(SimulateNetlist, WidthPopulationEdge[i], LengthPopulationEdge[i])
                UpdateParamsOfDelay(SimulateNetlist, WidthPopulationDelay[i], LengthPopulationDelay[i])


            # Accept the global condition
            if FitnessScore[i] < AccuracyThreshold:
                UpdateParamsOfEdge(SimulateNetlist, WidthPopulationEdge[i], LengthPopulationEdge[i])
                UpdateParamsOfDelay(SimulateNetlist, WidthPopulationDelay[i], LengthPopulationDelay[i])
                break




    # Plot the evolution of the best fitness score over generations
    plt.figure(figsize=(10, 6))
    plt.plot(FitnessHistory, marker='o', linestyle='-', color='b')
    plt.title('The Evolution of the Best Fitness Over the Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score')
    plt.grid(True)
    plt.savefig('NameOfGraph.png')

# Main function to set up and run the simulations
def main():
    # Define file paths and thresholds
    NetlistChain = 'Chain of inverter path .cir'  # Inverter chain path
    SimulateNetlist ='Simulate netlist path .cir' # Simulation cell path
    LowThreshold = 0.18
    HighThreshold = 1.62

    # Define simulation parameters for different experiments
    LowCapacity = 1
    HighCapacity = 60
    Toleration = 0.01

    PopSize = 10
    MaxGen = 200
    F = 0.8
    CR = 0.9
    WMin = 0.420
    WMax = 1.070
    LBase = 0.150
    LIncrease = 0.02
    AcThreshold = 1e-12
    ThreshLength = 5e-12
    Mode = " Type of cell "  #unbalanced/balanced

    # Define simulation steps and timing parameters
    SimParams = {
        'step_time': 0.05e-12,
        'end_time': 2.5e-9
    }

    # Call functions
    RiseTimeChain, FallTimeChain = SimulationChainOfInverters(NetlistChain, LowThreshold, HighThreshold, SimParams)
    ChangeParamsOfGen(RiseTimeChain, FallTimeChain, SimulateNetlist)
    TargetRiseTime = (RiseTimeChain + FallTimeChain) / 2
    SimulationCapacity(SimulateNetlist, SimParams, LowThreshold, HighThreshold)
    MeasCapacity(SimulateNetlist, SimParams, LowThreshold, HighThreshold, TargetRiseTime, LowCapacity, HighCapacity,
                 Toleration)
    SimulationTranzistors(SimulateNetlist, SimParams, LowThreshold, HighThreshold, Mode)

    # Run the differential evolution algorithm for optimization
    AlgorithmDE(SimulateNetlist, SimParams, LowThreshold, HighThreshold, PopSize,
                MaxGen, F, CR, WMin, WMax, LBase, LIncrease,
                AcThreshold, ThreshLength, Mode)

if __name__ == "__main__":
    main()
