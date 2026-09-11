from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

try:
    import torch
    sys.path.insert(0, str(Path(__file__).parents[1] / "tasks"))
    from zbot_rl_mjlab.mdp import body_shake_penalty, contact_point_slip, mechanical_power, small_step_penalty
except ImportError:
    torch = None
    contact_point_slip = None
    mechanical_power = None
    small_step_penalty = None
    body_shake_penalty = None


@unittest.skipIf(torch is None, "mjlab training runtime unavailable")
class ContactPointSlipTests(unittest.TestCase):
    def data(self, linear, angular):
        return SimpleNamespace(
            body_com_pos_w=torch.zeros((1, 2, 3)),
            body_com_lin_vel_w=torch.tensor([[linear, [0.0, 0.0, 0.0]]]),
            body_com_ang_vel_w=torch.tensor([[angular, [0.0, 0.0, 0.0]]]),
        )

    def sensor(self):
        return SimpleNamespace(
            found=torch.tensor([[1.0, 0.0]]),
            force=torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]]),
            pos=torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]),
            normal=torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]),
        )

    def test_rotation_at_contact_cancels_com_translation(self):
        card = SimpleNamespace(slip_force_threshold=1.0, slip_huber_delta=0.08)
        value = contact_point_slip(self.data([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]), self.sensor(), [0, 1], card)
        self.assertAlmostEqual(value.item(), 0.0, places=7)

    def test_tangential_contact_speed_uses_force_weighted_huber(self):
        card = SimpleNamespace(slip_force_threshold=1.0, slip_huber_delta=0.08)
        value = contact_point_slip(self.data([1.0, 0.0, 0.0], [0.0, 0.0, 2.0]), self.sensor(), [0, 1], card)
        self.assertAlmostEqual(value.item(), 0.96, places=6)

    def test_total_mechanical_power_integrates_absolute_joint_work(self):
        data = SimpleNamespace(qfrc_actuator=torch.tensor([[2.0, -3.0, 4.0]]), joint_vel=torch.tensor([[.5, 2.0, -1.0]]))
        self.assertAlmostEqual(mechanical_power(data).item(), 11.0, places=7)

    def test_small_step_combines_touchdown_distance_and_duration_shortfall(self):
        card = SimpleNamespace(moving_direction=1.0, minimum_step_advancement=.025, minimum_swing_duration=.08)
        value = small_step_penalty(torch.tensor([[.01, .03]]), torch.tensor([[.05, .1]]), torch.tensor([[True, True]]), card)
        self.assertAlmostEqual(value.item(), .975, places=6)

    def test_body_shake_ignores_forward_speed_and_reset_frame(self):
        angular = torch.tensor([[1.0, 2.0, 0.0]]); previous = torch.zeros_like(angular)
        linear = torch.tensor([[8.0, 1.0, 2.0]]); previous_linear = torch.tensor([[-8.0, 0.0, 0.0]])
        self.assertAlmostEqual(body_shake_penalty(angular, previous, linear, previous_linear, torch.tensor([False])).item(), 55.0, places=6)
        self.assertEqual(body_shake_penalty(angular, previous, linear, previous_linear, torch.tensor([True])).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
