"""Checkpoint/resume support for interrupted data collection runs."""

import json
import os
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Save and restore collection progress so interrupted runs can resume."""

    def __init__(self, checkpoint_dir: str, task_name: str):
        self.checkpoint_file = os.path.join(checkpoint_dir, f"{task_name}.json")
        self.state = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            logger.info(f"Resumed checkpoint: {self.checkpoint_file} ({len(state.get('completed', []))} items done)")
            return state
        return {"completed": [], "data": {}}

    def save(self):
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def is_done(self, key: str) -> bool:
        return key in self.state["completed"]

    def mark_done(self, key: str, data: dict | list | None = None):
        if key not in self.state["completed"]:
            self.state["completed"].append(key)
        if data is not None:
            self.state["data"][key] = data
        self.save()

    def get_data(self, key: str):
        return self.state["data"].get(key)

    def get_all_data(self) -> dict:
        return self.state["data"]

    def clear(self):
        self.state = {"completed": [], "data": {}}
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
