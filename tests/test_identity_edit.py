import os
import unittest

from tests.test_max_sample import _load_script


IDENTITY_SCRIPTS = [
    "firered_edit_changeperson.py",
    "flux2_klein_edit_changeperson.py",
]


def _sample():
    return {
        "color": ["黑色"],
        "clothing": ["衬衫"],
        "posture": ["正面站立"],
        "location": ["摄影棚"],
        "environmental_elements": ["纯色背景"],
        "lighting": ["柔和光线"],
        "hair_type_1": ["长直发"],
        "hair_type_2": ["短卷发"],
        "age": ["老年"],
        "gender": ["女性"],
        "ethnicity": ["东亚人"],
    }


class IdentityEditTest(unittest.TestCase):
    def test_output_dir_includes_identity_mode(self):
        for script_name in IDENTITY_SCRIPTS:
            with self.subTest(script=script_name):
                module = _load_script(script_name)

                self.assertTrue(module.OUTPUT_DIR.endswith(f"_{module.ID_EDIT_MODE}"))

    def test_identity_mode_can_come_from_environment(self):
        previous = os.environ.get("ID_EDIT_MODE")
        os.environ["ID_EDIT_MODE"] = "gender"
        try:
            for script_name in IDENTITY_SCRIPTS:
                with self.subTest(script=script_name):
                    module = _load_script(script_name)

                    self.assertEqual(module.ID_EDIT_MODE, "gender")
                    self.assertTrue(module.OUTPUT_DIR.endswith("_gender"))
        finally:
            if previous is None:
                os.environ.pop("ID_EDIT_MODE", None)
            else:
                os.environ["ID_EDIT_MODE"] = previous

    def test_age_prompt_changes_age_and_keeps_hair(self):
        for script_name in IDENTITY_SCRIPTS:
            with self.subTest(script=script_name):
                module = _load_script(script_name)
                prompt, negative_prompt = module.build_prompt(0, _sample(), edit_mode="age")

                self.assertIn("年龄阶段改为老年", prompt)
                self.assertIn("保持输入图中人物的发型", prompt)
                self.assertIn("不要改变发型", prompt)
                self.assertIn("改变发型", negative_prompt)

    def test_gender_prompt_changes_gender_and_keeps_hair(self):
        for script_name in IDENTITY_SCRIPTS:
            with self.subTest(script=script_name):
                module = _load_script(script_name)
                prompt, _negative_prompt = module.build_prompt(0, _sample(), edit_mode="gender")

                self.assertIn("性别表达改为女性", prompt)
                self.assertIn("保持输入图中人物的发型", prompt)

    def test_ethnicity_prompt_changes_ethnicity_and_keeps_hair(self):
        for script_name in IDENTITY_SCRIPTS:
            with self.subTest(script=script_name):
                module = _load_script(script_name)
                prompt, _negative_prompt = module.build_prompt(0, _sample(), edit_mode="ethnicity")

                self.assertIn("具有东亚人外貌特征", prompt)
                self.assertIn("保持输入图中人物的发型", prompt)

    def test_invalid_identity_mode_raises(self):
        for script_name in IDENTITY_SCRIPTS:
            with self.subTest(script=script_name):
                module = _load_script(script_name)

                with self.assertRaises(ValueError):
                    module.build_prompt(0, _sample(), edit_mode="unknown")


if __name__ == "__main__":
    unittest.main()
