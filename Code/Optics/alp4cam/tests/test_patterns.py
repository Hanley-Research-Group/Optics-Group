import numpy as np
import pytest
from PIL import Image

from alp4cam.core.config import AppConfig, load_config
from alp4cam.core.config import AppConfig as CoreAppConfig
from alp4cam.core.exceptions import ConfigurationError, PatternError, guarded
from alp4cam.dmd import DMDController
from alp4cam.dmd.controller import DMDController as RefactoredDMDController
from alp4cam.patterns import discover_pattern_files, generate_hadamard_set, generate_test_patterns, save_pattern_set
from alp4cam.cli.make_patterns import main as patterns_main


def test_refactored_module_layout_is_importable():
    assert CoreAppConfig is AppConfig
    assert RefactoredDMDController is DMDController


def test_generate_test_patterns():
    patterns = generate_test_patterns(4)
    assert len(patterns) == 4
    assert all(pattern.shape == (4, 4) for pattern in patterns)
    assert np.array_equal(patterns[0], np.zeros((4, 4), dtype=np.uint8))
    assert np.array_equal(patterns[1], np.full((4, 4), 255, dtype=np.uint8))


def test_generate_hadamard_set_has_expected_size():
    patterns = generate_hadamard_set(mode_side=32, macro=2)
    assert len(patterns) == 2048
    assert all(pattern.shape == (32, 32) for pattern in patterns)
    assert set(np.unique(patterns[0])).issubset({0, 255})


def test_discover_pattern_files(tmp_path):
    pattern_dir = tmp_path / "patterns"
    pattern_dir.mkdir()
    (pattern_dir / "a.png").write_bytes(b"x")
    (pattern_dir / "b.png").write_bytes(b"y")
    discovered = discover_pattern_files(pattern_dir)
    assert [p.name for p in discovered] == ["a.png", "b.png"]


def test_save_pattern_set_writes_pngs(tmp_path):
    patterns = [np.zeros((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.uint8) * 255]
    saved = save_pattern_set(patterns, tmp_path / "out", prefix="img")
    assert len(saved) == 2
    assert all(path.suffix == ".png" for path in saved)

    reloaded = [np.array(Image.open(path).convert("L")) for path in saved]
    assert set(np.unique(reloaded[0])).issubset({0})
    assert set(np.unique(reloaded[1])).issubset({0, 255})


def test_load_config_reads_default_yaml():
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.logging.log_level == "INFO"
    assert config.DMD.version == "4.2"


def test_load_config_rejects_unknown_keys():
    with pytest.raises(ConfigurationError):
        config = {"logging": {"log_level": "INFO", "extra": True}}
        from alp4cam.core.config import _coerce_config
        _coerce_config(config)


def test_dmd_controller_loads_pattern_sequence():
    controller = DMDController(version="4.2")
    patterns = [np.zeros((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.uint8) * 255]
    count = controller.load_patterns(patterns, picture_time_us=5000, once=False, duration_s=0)
    assert count == 2
    assert controller.device is not None
    assert controller.device.seq_alloc == (2, 1)
    assert controller.device.timing["pictureTime"] == 5000
    assert controller.device.run_loop is True


def test_cli_make_patterns_generates_files(tmp_path):
    output = tmp_path / "generated"
    result = patterns_main(["hadamard", "--out", str(output), "--mode-side", "16", "--macro", "2"])
    assert result == 0
    assert len(list(output.glob("*.png"))) == 2 * 16 * 16


def test_cli_make_patterns_emits_log_messages(tmp_path, capsys):
    output = tmp_path / "generated"
    result = patterns_main(["hadamard", "--out", str(output), "--mode-side", "8", "--macro", "1"])

    captured = capsys.readouterr()
    assert result == 0
    assert "Generated" in captured.err
    assert "Hadamard pattern set" in captured.err


def test_cli_make_patterns_creates_default_pattern_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = patterns_main(["hadamard", "--mode-side", "8", "--macro", "1"])
    assert result == 0
    folders = list((tmp_path / "data" / "patterns").glob("hadamard_*"))
    assert folders
    assert any(folder.glob("*.png") for folder in folders)


def test_guarded_exit_code_for_pattern_errors():
    @guarded
    def boom():
        raise PatternError("bad input")

    with pytest.raises(SystemExit) as exc:
        boom()
    assert exc.value.code == 4
