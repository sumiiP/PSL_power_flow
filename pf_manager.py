from modules.data_loader import DataLoader
from modules.pf_solver import PowerFlowLoader
from modules.exporter import Exporter

from config import *

class FlowManager : 
    def run_power_flow(self) : 
        # data load
        loader = DataLoader(BUS_FILE_PATH, LINE_FILE_PATH, BUS_FILE_HEADER_LIST, LINE_FILE_HEADER_LIST)
        loader.load_all_data()
        
        id_info = loader.distinguish_input_unknown_data()
        loader.convert_to_rectangular() # convert to rectangular coodinates
        
        y_bus = loader.make_admittance_matrix()

        # calculate the power flow equation
        solver = PowerFlowLoader(y_bus, loader.bus_df, id_info, loader.line_df)
        value = solver.calculate_power(MAX_ITER, TOLERANCE, LOAD_FACTOR, DAMPING_FACTOR)
        print(f"Convergence status: {value}")
        
        # data processing and results saving
        bus_results, summary_results = solver.get_final_results(ENERGY_BALANCE_TOLERANCE)
        
        # export to csv and graph
        exporter = Exporter(CSV_OUTPUT_PATH, GRAPH_OUTPUT_PATH, bus_results, summary_results)
        exporter.export_to_csv()
        exporter.export_to_graph()