from eda.DataManager import DataManager
from event_generator.SyntheticEventGenerator import SyntheticEventGenerator
from experiments.ThompsonSamplingSimulator import ThompsonSamplingSimulator

class CLI:
    def __init__(self):
        self.data_manager = DataManager()
        self.event_generator = SyntheticEventGenerator()
        self.thompson_sampling = ThompsonSamplingSimulator()

    def run_eda(self):
        self.data_manager.run_eda()

    def generate_events(self):
        self.event_generator.generate_events()

    def run_thompson_sampling(self):
        self.thompson_sampling.run_thompson_sampling()
