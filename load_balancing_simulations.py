from load_balancing_simulations.HeavyTraffic import SimulationHT
from load_balancing_simulations.MeanField import SimulationMF
from load_balancing_simulations.HalfinWitt import SimulationHW

# sim = SimulationHT(2, 0.95, 5 * 10 ** 6)
#
# sim.run_simulation()
# sim.draw_plots()
#
# sim = SimulationMF(100, 0.5, 10 ** 6)
#
# sim.run_simulation()
# sim.draw_plots()

sim = SimulationHW(50, 10 ** 6)

sim.run_simulation()
sim.draw_plots()
