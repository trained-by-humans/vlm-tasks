"""Check the contents of the actual wheel and source archive before publication."""
from pathlib import Path
import sys
import tarfile
import zipfile


def check_distribution(directory: Path) -> None:
    wheel, = directory.glob("*.whl")
    sdist, = directory.glob("*.tar.gz")
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = {item.filename for item in archive.infolist() if not item.is_dir()}
    with tarfile.open(sdist) as archive:
        prefix = sdist.name.removesuffix(".tar.gz") + "/"
        members = [item.name for item in archive.getmembers() if not item.isdir()]
        if any(not name.startswith(prefix) for name in members):
            raise ValueError(f"{sdist.name}: files outside the source archive root")
        source_names = {name.removeprefix(prefix) for name in members}

    for artifact, names, package in (
        (wheel, wheel_names, "ml_pipes/vlm/"),
        (sdist, source_names, "src/ml_pipes/vlm/"),
    ):
        required = {package + name for name in (
            "__init__.py", "validation/__init__.py", "validation/NOTICE.caption",
        )}
        metadata = {"LICENSE", "README.md", "pyproject.toml", "PKG-INFO"} if artifact == sdist else set()
        required |= metadata
        if artifact == sdist:
            # Hatchling retains VCS ignore rules for builds from the source archive.
            metadata.add(".gitignore")
        unexpected = {
            name for name in names
            if not name.startswith(package) and name not in metadata
            and not (artifact == wheel and name.split("/")[0].endswith(".dist-info"))
        }
        missing = required - names
        if unexpected or missing:
            raise ValueError(
                f"{artifact.name}: unexpected files {sorted(unexpected)}; "
                f"missing files {sorted(missing)}"
            )
        print(f"{artifact.name}: package contents verified ({len(names)} files)")


if __name__ == "__main__":
    check_distribution(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist"))
