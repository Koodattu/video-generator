from __future__ import annotations

from pathlib import Path
from video_generator import setup
from video_generator.runners import decode_wsl_output


def test_decode_wsl_output_accepts_redirected_utf16le() -> None:
    raw = "Ubuntu\r\nDebian\r\n".encode("utf-16-le")

    assert decode_wsl_output(raw).splitlines() == ["Ubuntu", "Debian"]


def test_native_cuda_runner_uses_automatic_torch_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    commands: list[list[str]] = []
    definition = setup.LOCAL_DEFINITIONS["local:flux.2-klein-4b"]

    def fake_run(command, *, cwd, environment=None, timeout=7200) -> str:
        values = [str(value) for value in command]
        commands.append(values)
        if values[1:] == ["--version"]:
            return "uv 0.6.9"
        if values[1] == "venv":
            python = setup._native_python(
                tmp_path / ".cache" / "runtimes" / setup.runner_slug(definition.backend_id)
            )
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_bytes(b"")
        return ""

    monkeypatch.setattr(setup, "_find_uv", lambda: "uv")
    monkeypatch.setattr(setup, "_run", fake_run)

    setup._install_native_environment(tmp_path, definition)

    compile_command = next(values for values in commands if values[1:3] == ["pip", "compile"])
    sync_command = next(values for values in commands if values[1:3] == ["pip", "sync"])
    assert compile_command[compile_command.index("--torch-backend") + 1] == "auto"
    assert sync_command[sync_command.index("--torch-backend") + 1] == "auto"


def test_parse_uv_version() -> None:
    assert setup._parse_uv_version("uv 0.6.9 (build metadata)") == (0, 6, 9)
