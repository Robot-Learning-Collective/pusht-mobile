"""End-to-end: browser recorder -> collector -> LeRobot -> Python replay.

Requires npm install, npm run build:wasm and the Python export dependencies.
Run: python -m unittest discover -s tests -v
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from server import main as server
from scripts.check_recordings import replay
from scripts.export_lerobot import state

ROOT = Path(__file__).resolve().parents[1]


class RecordingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.path = Path(cls.tmp.name)
        subprocess.run([str(ROOT / "node_modules/.bin/tsc"), "src/recording.ts", "--outDir", str(cls.path),
                        "--target", "es2022", "--module", "es2022", "--skipLibCheck"], cwd=ROOT, check=True)
        (cls.path / "recording.js").rename(cls.path / "recording.mjs")
        subprocess.run(["node", "tests/recording.mjs", str(cls.path / "recording.mjs"), str(cls.path / "fixtures.json")],
                       cwd=ROOT, check=True)
        cls.fixtures = json.loads((cls.path / "fixtures.json").read_text())

    def test_wasm_to_python_replay(self):
        for ep in self.fixtures["episodes"]:
            with self.subTest(seed=ep["init"]["seed"]):
                self.assertLessEqual(replay(ep), 1e-3)

    def test_collector_rejects_missing_state_or_bad_timing(self):
        with TestClient(server.app) as client:
            for change in ("missing_block", "gap", "nan", "legacy", "unsolved", "missing_final"):
                ep = copy.deepcopy(self.fixtures["solved"])
                if change == "missing_block":
                    del ep["frames"][0]["block"]
                elif change == "gap":
                    ep["frames"][1]["timestamp"] += 0.1
                elif change == "nan":
                    ep["frames"][0]["action"][0] = float("nan")
                elif change == "legacy":
                    del ep["schemaVersion"]
                elif change == "unsolved":
                    ep["success"] = False
                else:
                    del ep["finalState"]
                with self.subTest(change=change):
                    response = client.post("/api/episodes", content=json.dumps(ep), headers={"content-type": "application/json"})
                    self.assertEqual(response.status_code, 422)

    def test_save_export_and_replay(self):
        episodes = self.path / "episodes"
        episodes.mkdir()
        out = self.path / "export"
        ep = self.fixtures["solved"]
        with patch.object(server, "EPISODES_DIR", episodes), TestClient(server.app) as client:
            for i in range(5):
                response = client.post("/api/episodes", json=ep)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["total"], i + 1)
        subprocess.run([sys.executable, "scripts/export_lerobot.py", "--episodes", str(episodes), "--out", str(out),
                        "--repo-id", "test/recordings", "--state-only"], cwd=ROOT, check=True,
                       env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(ROOT / "python")})
        df = pd.concat(pd.read_parquet(p) for p in sorted((out / "data").rglob("*.parquet")))
        self.assertEqual(len(df), 5 * ep["nFrames"])
        self.assertNotIn("observation.image", df)
        for _, rows in df.groupby("episode_index"):
            np.testing.assert_array_equal(np.stack(rows["observation.state"]), [state(fr) for fr in ep["frames"]])
            np.testing.assert_array_equal(np.stack(rows["action"]), [fr["action"] for fr in ep["frames"]])
            np.testing.assert_allclose(rows["timestamp"], [fr["timestamp"] for fr in ep["frames"]], atol=1e-7)
            np.testing.assert_array_equal(np.stack(rows["next.observation.state"])[-1], state(ep["finalState"]))
            np.testing.assert_array_equal(np.asarray(rows["next.done"].tolist()).flatten(), [False, False, True])
        self.assertEqual(len(json.loads((out / "meta/recordings.json").read_text())), 5)
        for file in episodes.glob("*.json"):
            self.assertEqual(file.read_bytes(), (out / "raw" / file.name).read_bytes())
            replay(json.loads(file.read_text()))


if __name__ == "__main__":
    unittest.main()
