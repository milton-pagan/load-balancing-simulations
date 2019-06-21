from load_balancing_simulations.HalfinWitt import SimulationHW
from load_balancing_simulations.HeavyTraffic import SimulationHT
from load_balancing_simulations.MeanField import SimulationMF
from load_balancing_simulations.gui.main_window import MainWindow

main_window = MainWindow()
select, n, lambda_, arrivals = main_window.run()

# Run selected simulation
if select == 'ht':

    sim = SimulationHT(n, lambda_, arrivals)
    sim.run_simulation()
    sim.draw_plots()

elif select == 'mf':

    sim = SimulationMF(n, lambda_, arrivals)
    sim.run_simulation()
    sim.draw_plots()

elif select == 'hw':

    sim = SimulationHW(n, arrivals)
    sim.run_simulation()
    sim.draw_plots()
