"""Offline round-trip tests for the pure core (no GPU / no Wan2GP).

Run:  python tests/test_core.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image

from Replicant.core import character, datasets, poses


def _img(path, color):
    Image.new("RGB", (64, 96), color).save(path)


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        base = d / "base_src.png"; _img(base, (200, 50, 50))
        pose1 = d / "p1.png"; _img(pose1, (50, 200, 50))
        cs = character.CharacterState(
            name="Nova Test", description="a woman with red hair",
            style="realism", positive_prompt="pos", negative_prompt="neg",
            selected_base=str(base), approved_poses=[str(pose1)],
            approved_pose_specs=[{"distance": "full", "angle": "side"}])
        cdir = d / "char"
        character.save_character(cdir, cs)
        assert (cdir / "character.json").is_file()
        assert (cdir / "base.png").is_file()
        saved_pose = list((cdir / "poses").glob("*.png"))
        assert saved_pose and saved_pose[0].name == "pose_001__full__side.png", saved_pose

        cs2 = character.load_character(cdir)
        assert cs2.name == "Nova Test"
        assert cs2.description == "a woman with red hair"
        assert cs2.trigger == "nova_test"
        assert character.parse_pose_distance_angle(saved_pose[0]) == ("full", "side")
    print("✓ save/load round-trip")


def test_dataset_compose():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ps = []
        for i in range(4):
            p = d / f"pose{i}.png"; _img(p, (i * 40, 100, 150)); ps.append(str(p))
        cs = character.CharacterState(name="Cmp", description="desc", approved_poses=ps,
            approved_pose_specs=[{"distance": "full", "angle": "front"}] * 4)
        # fake detector -> no faces, so head/upper crops are skipped (offline)
        class _NoFaces:
            def detect_faces(self, _): return []
        out = datasets.build_character_datasets(d / "char", cs, detector=_NoFaces())
        assert out["trigger"] == "cmp"
        for key in ("video512", "highres", "full"):
            files = list(Path(out[key]).glob("*.png"))
            caps = list(Path(out[key]).glob("*.txt"))
            assert files and caps, (key, files, caps)
            sample = caps[0].read_text()
            assert sample.startswith("cmp,"), sample
    print("✓ dataset compose + captions")


def test_poses_and_negatives():
    assert len(poses.POSES) == 21
    assert all(p.distance in ("close", "medium", "full") for p in poses.POSES)
    assert "close-up" in poses.pose_negative_for("full", "x")
    assert "close-up" not in poses.pose_negative_for("close", "x")
    print("✓ pose specs + distance-conditional negatives")


def test_seed_prompt():
    p = character.build_seed_prompt("a woman with red hair", "realism")
    assert p.startswith("photo,"), p
    assert "a woman with red hair" in p
    assert "full body" in p
    assert character.build_seed_prompt("x", "anime").startswith("anime,")
    assert character.DEFAULT_NEGATIVE  # non-empty
    print("✓ seed prompt builder + style medium")


if __name__ == "__main__":
    test_save_load_roundtrip()
    test_dataset_compose()
    test_poses_and_negatives()
    test_seed_prompt()
    print("\nALL PASSED")
