from HeavyTraffic import SimulationHT
from MeanField import SimulationMF
from HalfinWitt import SimulationHW

# sim = SimulationHT(2, 0.95, 5 * 10 ** 6)
#
# sim.run_simulation()
# sim.draw_plots()

# sim = SimulationMF(100, 0.5, 10 ** 6)
#
# sim.run_simulation()
# sim.draw_plots()

sim = SimulationHW(500, 10 ** 6)

sim.run_simulation()
sim.draw_plots()
