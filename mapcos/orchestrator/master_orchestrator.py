"""
Master Orchestrator
-------------------
Receives one patient case and dispatches the agents.

Stage 1 (parallel): CNN Classification Agent + Follicle Counting Agent on the
                    ultrasound image.
Stage 2           : Grounding Agent cross-checks the two vision outputs.
Stage 3 (parallel): Lab Agent + Symptoms Agent on the tabular data.
Stage 4           : Diagnostic Synthesis Agent (Rotterdam 2-of-3 rule), then the
                    Exclusion Agent (thyroid / prolactin check).

Agents are plain threads (ThreadPoolExecutor); no agent framework is used. The lab,
symptoms, synthesis and exclusion agents are optional constructor arguments: when one
is None its stage is skipped and the result stays None.
"""

from concurrent.futures import ThreadPoolExecutor


class MasterOrchestrator:
    def __init__(self, cnn_agent, follicle_agent, grounding_agent,
                 lab_agent=None, symptoms_agent=None,
                 synthesis_agent=None, exclusion_agent=None):
        self.cnn_agent = cnn_agent
        self.follicle_agent = follicle_agent
        self.grounding_agent = grounding_agent
        self.lab_agent = lab_agent
        self.symptoms_agent = symptoms_agent
        self.synthesis_agent = synthesis_agent
        self.exclusion_agent = exclusion_agent

    def _run_vision_agents(self, image_path: str) -> dict:
        with ThreadPoolExecutor(max_workers=2) as executor:
            cnn_future = executor.submit(self.cnn_agent.run, image_path)
            follicle_future = executor.submit(self.follicle_agent.run, image_path)
            return {
                "cnn": cnn_future.result(),
                "follicle": follicle_future.result(),
            }

    def _run_tabular_agents(self, tabular_data: dict) -> dict:
        results = {"lab": None, "symptoms": None}
        agents_to_run = {}
        if self.lab_agent is not None:
            agents_to_run["lab"] = (self.lab_agent.run, tabular_data.get("lab_values", {}))
        if self.symptoms_agent is not None:
            agents_to_run["symptoms"] = (self.symptoms_agent.run, tabular_data.get("symptoms", {}))

        if not agents_to_run:
            return results

        with ThreadPoolExecutor(max_workers=len(agents_to_run)) as executor:
            futures = {name: executor.submit(fn, arg) for name, (fn, arg) in agents_to_run.items()}
            for name, future in futures.items():
                results[name] = future.result()
        return results

    def run(self, patient_input: dict) -> dict:
        """
        patient_input:
            {
                "image_path": "<path to ultrasound image>",
                "tabular_data": {                      # optional
                    "lab_values": {"fsh": ..., "lh": ..., "amh": ...},
                    "symptoms": {"bmi": ..., "cycle_irregular": 0/1, "hair_growth": 0/1,
                                 "weight_gain": 0/1, "skin_darkening": 0/1,
                                 "hair_loss": 0/1, "pimples": 0/1},
                    "exclusion": {"tsh": ..., "prl": ...},   # used by the Exclusion Agent
                }
            }
        """
        vision = self._run_vision_agents(patient_input["image_path"])
        grounding_result = self.grounding_agent.run(vision["cnn"], vision["follicle"])

        result = {
            "cnn_classification": vision["cnn"],
            "follicle_counting": vision["follicle"],
            "grounding": grounding_result,
            "lab": None,
            "symptoms": None,
            "synthesis": None,
            "exclusion": None,
        }

        tabular_data = patient_input.get("tabular_data")
        if tabular_data and (self.lab_agent or self.symptoms_agent):
            tabular = self._run_tabular_agents(tabular_data)
            result["lab"] = tabular["lab"]
            result["symptoms"] = tabular["symptoms"]

        # --- Diagnostic Synthesis (Rotterdam 2-of-3) and Exclusion ---------------
        if self.synthesis_agent is not None:
            td = tabular_data or {}
            result["synthesis"] = self.synthesis_agent.run(
                vision["cnn"], vision["follicle"], grounding_result,
                result["lab"], result["symptoms"], td.get("symptoms", {}))
            if self.exclusion_agent is not None:
                result["exclusion"] = self.exclusion_agent.run(
                    result["synthesis"], td.get("exclusion", {}))
        # -------------------------------------------------------------------------

        return result
