import unittest
from unittest import mock

from detect import hardware_power as hp


class DeviceClassifyTests(unittest.TestCase):
    def test_families(self):
        self.assertEqual(hp.classify_device_model("NVIDIA Jetson Orin Nano"), "jetson")
        self.assertEqual(hp.classify_device_model("Raspberry Pi 5 Model B"), "raspberry_pi_5")
        self.assertEqual(hp.classify_device_model("Raspberry Pi 4 Model B"), "raspberry_pi_4")
        self.assertEqual(hp.classify_device_model("Some Laptop"), "other")


class TegrastatsParseTests(unittest.TestCase):
    def test_vdd_in_milliwatts(self):
        self.assertEqual(hp._parse_tegrastats_power("RAM 1/2 VDD_IN 2532mW/2532mW"), 2.532)

    def test_pom_5v_in_plain_milliwatts(self):
        self.assertEqual(hp._parse_tegrastats_power("POM_5V_IN 4286/4286"), 4.286)

    def test_orin_nano_has_no_power_rails(self):
        # Orin Nano tegrastats: no VDD_*/POM_* power fields -> None
        self.assertIsNone(hp._parse_tegrastats_power(
            "RAM 900/7620MB SWAP 0/3810MB CPU [1%@1420] GR3D_FREQ 0%"))


class PiPmicParseTests(unittest.TestCase):
    def test_sum_of_v_times_i(self):
        sample = "\n".join([
            "VDD_CORE_V volt(24)=0.75V",
            "VDD_CORE_A current(7)=2.0A",   # 1.5 W
            "3V7_WL_SW_V volt(6)=3.70V",
            "3V7_WL_SW_A current(8)=0.5A",  # 1.85 W
        ])
        completed = mock.Mock(returncode=0, stdout=sample)
        with mock.patch.object(hp.subprocess, "run", return_value=completed):
            self.assertEqual(hp._read_pi_power_watts(), round(1.5 + 1.85, 2))

    def test_vcgencmd_missing_returns_none(self):
        with mock.patch.object(hp.subprocess, "run", side_effect=FileNotFoundError):
            self.assertIsNone(hp._read_pi_power_watts())


class ExternalMeterTests(unittest.TestCase):
    def test_parses_watts_from_command_output(self):
        completed = mock.Mock(returncode=0, stdout="5.42 W\n")
        with mock.patch.dict("os.environ", {"PILLBOX_POWER_METER_CMD": "meter"}), \
                mock.patch.object(hp.subprocess, "run", return_value=completed):
            self.assertEqual(hp._read_external_meter_watts(), 5.42)

    def test_unset_env_returns_none(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(hp._read_external_meter_watts())

    def test_nonzero_exit_returns_none(self):
        completed = mock.Mock(returncode=1, stdout="")
        with mock.patch.dict("os.environ", {"PILLBOX_POWER_METER_CMD": "meter"}), \
                mock.patch.object(hp.subprocess, "run", return_value=completed):
            self.assertIsNone(hp._read_external_meter_watts())


class SourcePriorityTests(unittest.TestCase):
    def test_external_meter_wins_and_reports_source(self):
        with mock.patch.object(hp, "_read_external_meter_watts", return_value=6.0), \
                mock.patch.object(hp, "_read_pi_power_watts", return_value=3.0), \
                mock.patch.object(hp, "_read_jetson_power_watts", return_value=None):
            self.assertEqual(hp.read_power_watts(with_source=True), (6.0, "external_meter"))

    def test_falls_through_to_none(self):
        with mock.patch.object(hp, "_read_external_meter_watts", return_value=None), \
                mock.patch.object(hp, "_read_pi_power_watts", return_value=None), \
                mock.patch.object(hp, "_read_jetson_power_watts", return_value=None):
            self.assertEqual(hp.read_power_watts(with_source=True), (None, None))
            self.assertIsNone(hp.read_power_watts())


if __name__ == "__main__":
    unittest.main()
