import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from detect import calibrate_camera, camera_profiles


class ProfileLoadingTests(unittest.TestCase):
    def test_imx708_default_loads_and_validates(self):
        p = camera_profiles.load_profile()  # env unset -> imx708
        self.assertEqual(p["name"], "imx708")
        self.assertEqual(p["ref_image"], "photo_20260713_142841.jpg")
        self.assertEqual(len(p["ref_quad"]), 4)
        self.assertEqual(len(p["days"]) * len(p["slots"]), 21)

    def test_env_var_selects_profile(self):
        with mock.patch.dict("os.environ", {"PILLBOX_CAMERA_PROFILE": "imx708"}):
            self.assertEqual(camera_profiles.load_profile()["name"], "imx708")

    def test_unknown_profile_is_actionable(self):
        with self.assertRaises(camera_profiles.ProfileError) as ctx:
            camera_profiles.load_profile("no_such_camera")
        self.assertIn("calibrate_camera.py", str(ctx.exception))
        self.assertIn("Available:", str(ctx.exception))

    def test_missing_key_rejected(self):
        bad = {"ref_image": "x.jpg"}  # everything else missing
        with self.assertRaises(camera_profiles.ProfileError) as ctx:
            camera_profiles._validate("bad", bad)
        self.assertIn("missing keys", str(ctx.exception))

    def test_bad_quad_rejected(self):
        good = camera_profiles.load_profile()
        good = dict(good)
        good["ref_quad"] = [[0, 0], [1, 1]]  # only 2 points
        with self.assertRaises(camera_profiles.ProfileError):
            camera_profiles._validate("bad", good)

    def test_match_scale_range_enforced(self):
        good = dict(camera_profiles.load_profile())
        good["match_scale"] = 2.0
        with self.assertRaises(camera_profiles.ProfileError):
            camera_profiles._validate("bad", good)


class AnchorDerivationTests(unittest.TestCase):
    def test_anchors_cover_left_and_right_of_quad(self):
        quad = [[100, 50], [900, 50], [900, 450], [100, 450]]  # 800x400 box
        left, right = calibrate_camera.anchors_from_quad(quad, frac=0.44)
        # left strip starts at the left edge; right strip ends at the right edge
        self.assertEqual(left[0], 100)
        self.assertEqual(right[2], 900)
        # each spans ~44% of width
        self.assertAlmostEqual(left[2] - left[0], 0.44 * 800, delta=1)
        self.assertAlmostEqual(right[2] - right[0], 0.44 * 800, delta=1)
        # both full-height (with a little pad on the bottom)
        self.assertEqual(left[1], 50)
        self.assertGreaterEqual(left[3], 450)


class CalibrateCameraTests(unittest.TestCase):
    def test_writes_a_loadable_profile_from_a_quad(self):
        with TemporaryDirectory() as tmp:
            photo = Path(tmp) / "empty_jetson.jpg"
            photo.write_bytes(b"\xff\xd8\xff\xd9")  # stand-in file; not decoded
            out = Path(tmp) / "jetson_wyze.json"
            argv = [
                "calibrate_camera", "--photo", str(photo), "--name", "jetson_wyze",
                "--quad", "640,180", "1890,175", "1900,1010", "630,1015",
            ]
            with mock.patch.object(camera_profiles, "profile_path",
                                   return_value=out), \
                    mock.patch("sys.argv", argv):
                calibrate_camera.main()
            data = json.loads(out.read_text())
            camera_profiles._validate("jetson_wyze", data)  # must pass loader
            self.assertEqual(data["ref_image"], "empty_jetson.jpg")
            self.assertEqual(data["ref_quad"][0], [640, 180])
            self.assertEqual(len(data["anchors"]), 2)


if __name__ == "__main__":
    unittest.main()
