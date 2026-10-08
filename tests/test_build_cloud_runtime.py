import subprocess
from pathlib import Path

from scripts.build_cloud_runtime import apply_patch_once, sha256


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_patch_is_applied_once(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init")
    git(source, "config", "user.email", "test@example.invalid")
    git(source, "config", "user.name", "Test")
    target = source / "value.txt"
    target.write_text("before\n")
    git(source, "add", "value.txt")
    git(source, "commit", "-m", "base")
    target.write_text("after\n")
    patch = tmp_path / "change.patch"
    patch.write_bytes(
        subprocess.run(
            ["git", "diff", "--binary"], cwd=source, check=True, capture_output=True
        ).stdout
    )
    git(source, "checkout", "--", "value.txt")

    assert apply_patch_once(source, patch) == "applied"
    assert target.read_text() == "after\n"
    assert apply_patch_once(source, patch) == "already-applied"
    assert len(sha256(patch)) == 64
