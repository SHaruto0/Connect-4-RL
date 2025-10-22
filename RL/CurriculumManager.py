import json
import torch

class CurriculumManager:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)["curriculum"]
        self.stage = 0

    def get_current_stage(self):
        return self.config[self.stage]

    def advance(self):
        self.stage += 1

if __name__ == "__main__":
    manager = CurriculumManager("curriculum_config.json")

    for stage_cfg in manager.config:
        print(stage_cfg)
        mode = stage_cfg['mode']
        epi = stage_cfg['episodes']
        saveas = stage_cfg['saveas']

        print(mode, epi, saveas)