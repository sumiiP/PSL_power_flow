import pandas as pd
import matplotlib.pyplot as plt

import os
from datetime import datetime

class Exporter :
    def __init__(self, CSV_OUTPUT_PATH, GRAPH_OUTPUT_PATH, bus_results, summary_results) :
        self.CSV_OUTPUT_PATH = CSV_OUTPUT_PATH
        self.GRAPH_OUTPUT_PATH = GRAPH_OUTPUT_PATH
        self.bus_results = bus_results
        self.summary_results = summary_results
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories if they don't exist
        os.makedirs(self.CSV_OUTPUT_PATH, exist_ok=True)
        os.makedirs(self.GRAPH_OUTPUT_PATH, exist_ok=True)
        
    def export_to_csv(self) :
        file_name = f"result_{self.timestamp}.csv"
        file_path = os.path.join(self.CSV_OUTPUT_PATH, file_name)
        
        with open(file_path, "w", encoding="utf-8") as f :
            
            f.write("Bus Results\n")
            self.bus_results.to_csv(f, index=False, float_format="%.6f")
            
            f.write("\nSummary Results\n")
            self.summary_results.to_csv(f, index=False, float_format="%.6f")
            
        print(f"✅ Results exported to {file_path}")
        
    def export_to_graph(self) :
        file_name = f"graph_{self.timestamp}.png"
        file_path = os.path.join(self.GRAPH_OUTPUT_PATH, file_name)
        
        plt.figure(figsize=(10, 8))
        
        # Voltage Magnitude Plot
        plt.subplot(2, 1, 1)
        plt.plot(self.bus_results['BUS'], self.bus_results['|V| (p.u.)'], 'r-o', markersize=4)
        plt.title('Voltage Magnitude per Bus')
        plt.xlabel('BUS index num')
        plt.ylabel('|V| (p.u.)')
        plt.grid(True)
        
        # Voltage Angle Plot
        plt.subplot(2, 1, 2)
        plt.plot(self.bus_results['BUS'], self.bus_results['δ (deg)'], 'b-s', markersize=4)
        plt.title('Voltage Angle per Bus')
        plt.xlabel('BUS index num')
        plt.ylabel('δ (deg)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close
        
        print(f"✅ Graphs generated at {file_path}")