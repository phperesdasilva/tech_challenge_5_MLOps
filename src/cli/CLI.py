from eda.DataManager import DataManager
from event_generator.SyntheticEventGenerator import SyntheticEventGenerator
from experiments.ThompsonSamplingSimulator import ThompsonSamplingSimulator
from assistant.assistant import ExperimentAssistant


class CLI:
    def __init__(self):
        self.data_manager = DataManager()
        self.event_generator = SyntheticEventGenerator()
        self.thompson_sampling = ThompsonSamplingSimulator()
        self._assistant = None

    @property
    def assistant(self):
        if self._assistant is None:
            self._assistant = ExperimentAssistant()
        return self._assistant

    def run_eda(self):
        self.data_manager.run_eda()

    def generate_events(self):
        self.event_generator.generate_events()

    def run_thompson_sampling(self):
        self.thompson_sampling.run_thompson_sampling()

    def summarize(self):
        resp = self.assistant.summarize_experiment()
        print(resp.answer)

    def ask(self, question: str):
        resp = self.assistant.answer_policy_question(question)
        print(resp.answer)
        if resp.sources:
            print("\nFontes:", ", ".join(resp.sources))

    def explain(self, arm_id: str, reason_codes: list):
        resp = self.assistant.explain_decision({
            "arm_id": arm_id,
            "reason_codes": reason_codes,
            "context": {},
        })
        print(resp.answer)
