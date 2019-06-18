from Simulation_HT import SimulationHT
from Simulation_MF import SimulationMF
from Simulation_HW import SimulationHW

sim = SimulationHT(2, 0.95, 5 * 10 ** 6)

sim.run_simulation()
sim.draw_plots()

# sim = SimulationMF(100, 0.5, 10 ** 6)
#
# sim.run_simulation()
# sim.draw_plots()
#
# sim = SimulationHW(100, 10 ** 6)
#
# sim.run_simulation()
# sim.draw_plots()
