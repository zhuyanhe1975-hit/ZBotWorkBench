import importlib.util
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("training_worker", Path(__file__).parents[1] / "worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class RequestTests(unittest.TestCase):
    def test_live_frame_is_bounded_and_keeps_real_pose_values(self):
        class Tensor:
            def __init__(self, value): self.value = value
            @property
            def shape(self): return (len(self.value), len(self.value[0]), len(self.value[0][0]))
            def __getitem__(self, key): return Tensor(self.value[key])
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return self.value
        class Value: pass
        robot = Value(); robot.data = Value(); robot.body_names = ["base", "leg"]
        robot.data.body_link_pos_w = Tensor([[[i, 0, .25], [i, 0, .1]] for i in range(12)])
        robot.data.body_link_quat_w = Tensor([[[1, 0, 0, 0], [1, 0, 0, 0]] for _ in range(12)])
        origins = Tensor([[i, 0, 0] for i in range(12)])
        frame = worker.live_frame(robot, origins, 7)
        self.assertEqual(frame["iteration"], 7)
        self.assertEqual(len(frame["environments"]), 9)
        self.assertEqual(frame["environments"][2]["bodyPositions"][0], [2, 0, .25])
        self.assertEqual(frame["environments"][2]["environmentOrigin"], [2, 0, 0])

    def test_live_preview_defaults_off_and_uses_an_owned_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            self.assertFalse(worker.live_preview_enabled(job))
            (job / "LIVE_PREVIEW").touch()
            self.assertTrue(worker.live_preview_enabled(job))

    def test_reward_terms_are_aggregated_by_component_and_ignore_other_logs(self):
        values = worker.aggregate_reward_terms([
            {"Reward/walk": 2.0, "Reward/slide": -1.0, "Metrics/speed": 3},
            {"Reward/walk": 4.0, "Reward/slide": -3.0},
        ])
        self.assertEqual(values, {"walk": 3.0, "slide": -2.0})

    def request(self):
        return dict(taskId=worker.TASK_ID, device="cpu", cpuThreads=2, numEnvs=1,
                    iterations=1, saveInterval=1, seed=42, maxSeconds=60)

    def test_cpu_and_explicit_gpu(self):
        for device in ("cpu", "cuda:0", "cuda:2"):
            worker.validate_request({**self.request(), "device": device})

    def test_walking_preset_is_hash_verified_and_configuration_preserved(self):
        import json
        from types import SimpleNamespace
        worker.validate_request({**self.request(), "taskId": worker.WALKING_TASK_ID})
        checkpoint = worker.preset_checkpoint()
        config = json.loads(checkpoint.with_name("config.json").read_text())
        self.assertEqual(config["task_card"]["physics_dt"], 1 / 240)
        self.assertEqual(config["task_card"]["decimation"], 8)
        self.assertEqual(config["task_card"]["initial_stage"], 2)
        self.assertEqual(config["task_card"]["joint_speed_range"], [0.8, 1.2])
        self.assertEqual(config["ppo"]["algorithm"]["learning_rate"], 1e-5)
        self.assertEqual(config["ppo"]["algorithm"]["entropy_coef"], 0)
        state = {"actor_state_dict": {"mlp.0.weight": SimpleNamespace(shape=(128, 30))}}
        worker.validate_policy_config(config, state, worker.WALKING_TASK_ID)
        worker.validate_policy_config(config, state, worker.QUASISTATIC_TASK_ID)
        with self.assertRaises(ValueError):
            worker.validate_policy_config(config, state, worker.TASK_ID)
        config["ppo"]["critic"]["class_name"] = "untrusted.module:Model"
        with self.assertRaises(ValueError):
            worker.validate_policy_config(config, state, worker.WALKING_TASK_ID)

    def test_quasistatic_preset_uses_slow_target_and_safety_rewards(self):
        from dataclasses import dataclass
        @dataclass(frozen=True)
        class Card:
            joint_speed_range: tuple = (.8, 1.2)
            target_forward_speed: float = .5
            velocity_sigma: float = .3
            stage_2_rewards: dict = None
        card = worker.quasistatic_overrides(Card())
        self.assertEqual(card.joint_speed_range, (.2, .4))
        self.assertEqual(card.target_forward_speed, .04)
        self.assertEqual(card.velocity_sigma, .02)
        self.assertEqual(card.stage_2_rewards["support_stability"], 4)
        self.assertEqual(card.stage_2_rewards["double_flight"], -10)

    def test_from_scratch_uses_verified_physics_and_rejects_legacy_resume(self):
        from types import SimpleNamespace
        overrides = worker.from_scratch_overrides()
        self.assertEqual(overrides["physics_dt"], 1 / 240)
        self.assertEqual(overrides["decimation"], 8)
        self.assertEqual(overrides["contact_history_length"], 8)
        config = {"task_card": {"physics_dt": 1 / 60, "decimation": 2, "contact_history_length": 5},
                  "ppo": {"actor": {"class_name": "MLPModel", "activation": "elu", "obs_normalization": False, "rnn_type": None, "cnn_cfg": None},
                          "critic": {"class_name": "MLPModel", "activation": "elu", "obs_normalization": False, "rnn_type": None, "cnn_cfg": None},
                          "algorithm": {"class_name": "PPO"}}}
        state = {"actor_state_dict": {"mlp.0.weight": SimpleNamespace(shape=(128, 26))}}
        with self.assertRaisesRegex(ValueError, "60 Hz"):
            worker.validate_policy_config(config, state, worker.TASK_ID)

    def test_task_card_reward_weights_are_allowlisted_and_immutable(self):
        from dataclasses import dataclass, field
        @dataclass(frozen=True)
        class Card:
            stage_1_rewards: dict = field(default_factory=lambda: {"stand": 1.0})
            stage_2_rewards: dict = field(default_factory=lambda: {"walk": 2.0})
            terminated_reward_penalty: float = 20.0
        original = Card()
        updated = worker.apply_task_card_settings(original, {"stage1Rewards": {"stand": 3}, "stage2Rewards": {"walk": 4}, "terminatedRewardPenalty": 25})
        self.assertEqual(updated.stage_1_rewards, {"stand": 3})
        self.assertEqual(updated.terminated_reward_penalty, 25)
        self.assertEqual(original.stage_1_rewards, {"stand": 1.0})
        with self.assertRaises(ValueError):
            worker.apply_task_card_settings(original, {"stage1Rewards": {"other": 1}, "stage2Rewards": {"walk": 2}, "terminatedRewardPenalty": 20})

    def test_reject_invalid_resources(self):
        for key, value in (("cpuThreads", 0), ("numEnvs", 4097), ("iterations", True),
                           ("maxSeconds", 0), ("device", "cuda"), ("taskId", "other")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.validate_request({**self.request(), key: value})

    def test_cancelled_job_finishes_without_importing_training_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            (job / "STOP").touch()
            events = []
            worker.run(self.request(), job, lambda event, **fields: events.append({"event": event, **fields}))
            self.assertEqual(events, [{"event": "finished", "stopped": True, "iterations": 0}])

    def test_cpu_environment_limit_matches_service(self):
        worker.validate_request({**self.request(), "numEnvs": 64})
        with self.assertRaises(ValueError):
            worker.validate_request({**self.request(), "numEnvs": 65})
        worker.validate_request({**self.request(), "device": "cuda:1", "numEnvs": 65})

    def test_gpu_uses_requested_physical_index_not_inherited_mask(self):
        environment = {"CUDA_VISIBLE_DEVICES": "5,3", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}
        actual = worker.prepare_device("cuda:3", environment)
        self.assertEqual(actual, "cuda:0")
        self.assertEqual(environment, {"CUDA_VISIBLE_DEVICES": "3", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})

    def test_cpu_clears_inherited_cuda_devices(self):
        environment = {"CUDA_VISIBLE_DEVICES": "1,2"}
        self.assertEqual(worker.prepare_device("cpu", environment), "cpu")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")

    def test_default_gpu_zero_remains_gpu_zero(self):
        environment = {}
        self.assertEqual(worker.prepare_device("cuda:0", environment), "cuda:0")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")

    def test_affinity_uses_only_requested_count_inside_inherited_mask(self):
        with patch.object(worker.os, "sched_getaffinity", return_value={11, 5, 8}, create=True) as get_mask, \
                patch.object(worker.os, "sched_setaffinity", create=True) as set_mask:
            self.assertEqual(worker.apply_cpu_affinity(2), [5, 8])
            get_mask.assert_called_once_with(0)
            set_mask.assert_called_once_with(0, {5, 8})

    def test_affinity_never_expands_available_cpus(self):
        with patch.object(worker.os, "sched_getaffinity", return_value={4, 12}, create=True), \
                patch.object(worker.os, "sched_setaffinity", create=True) as set_mask:
            self.assertEqual(worker.apply_cpu_affinity(16), [4, 12])
            set_mask.assert_called_once_with(0, {4, 12})

    def test_affinity_failure_leaves_library_thread_limits_available(self):
        with patch.object(worker.os, "sched_getaffinity", return_value={4, 12}, create=True), \
                patch.object(worker.os, "sched_setaffinity", side_effect=OSError("unsupported"), create=True):
            self.assertIsNone(worker.apply_cpu_affinity(1))


if __name__ == "__main__":
    unittest.main()
