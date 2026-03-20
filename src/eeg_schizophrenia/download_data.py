from __future__ import annotations

import argparse
import inspect
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, Protocol, Sequence
from zipfile import ZipFile

DEFAULT_DATASET = "broach/button-tone-sz"
DEFAULT_OUTPUT_DIR = Path("data/raw")
AUTH_GUIDANCE = (
    "Set KAGGLE_API_TOKEN (recommended) or place the token in ~/.kaggle/access_token. "
    "Legacy kaggle.json and KAGGLE_USERNAME/KAGGLE_KEY also work."
)


class KaggleDatasetApi(Protocol):
    def authenticate(self) -> Any: ...

    def dataset_list_files(self, dataset: str, **kwargs: Any) -> Any: ...

    def dataset_download_file( self, dataset: str, file_name: str, **kwargs: Any,) -> Any: ...


@dataclass(slots=True)
class DownloadSummary:
    dataset: str
    output_dir: Path
    all_csv_paths: list[Path]
    downloaded_paths: list[Path]
    skipped_paths: list[Path]


def download_dataset_csvs(
    dataset: str = DEFAULT_DATASET,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
    api: KaggleDatasetApi | None = None,
    progress: bool = False,
) -> DownloadSummary:
    """Download only CSV files from a Kaggle dataset.

    The function lists remote dataset files first and downloads matching CSVs one
    at a time. That keeps `.tar` and other non-CSV assets out of the request path.
    """

    kaggle_api = api or _build_kaggle_api()
    _authenticate_api(kaggle_api)

    output_root = Path(output_dir)
    if progress: 
        print(f"Listing CSV files in {dataset}...")
    remote_csv_files = _list_remote_csv_files(kaggle_api, dataset)
    if not remote_csv_files:
        raise ValueError(f"No CSV files were found in Kaggle dataset '{dataset}'.")

    all_csv_paths: list[Path] = []
    downloaded_paths: list[Path] = []
    skipped_paths: list[Path] = []

    for remote_name in remote_csv_files:
        destination = output_root.joinpath(*PurePosixPath(remote_name).parts)
        all_csv_paths.append(destination)

        if destination.exists() and not force:
            if progress:
                print(f"Skipping: {remote_name} (already exists)")
            skipped_paths.append(destination)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if progress:
                print(f"Downloading: {remote_name}")
            _download_remote_csv(
                api=kaggle_api,
                dataset=dataset,
                remote_file_name=remote_name,
                destination=destination,
                force=force,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download '{remote_name}' from Kaggle dataset '{dataset}'."
            ) from exc
        downloaded_paths.append(destination)

    return DownloadSummary(
        dataset=dataset,
        output_dir=output_root,
        all_csv_paths=all_csv_paths,
        downloaded_paths=downloaded_paths,
        skipped_paths=skipped_paths,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download CSV files from a Kaggle dataset."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Kaggle dataset slug.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR.as_posix(),
        help="Directory where CSV files will be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download CSV files even if they already exist locally.",
    )
    args = parser.parse_args(argv)

    try:
        summary = download_dataset_csvs(
            dataset=args.dataset,
            output_dir=args.output_dir,
            force=args.force,
            progress=True,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        f"Dataset '{summary.dataset}': downloaded {len(summary.downloaded_paths)} CSV file(s), "
        f"skipped {len(summary.skipped_paths)} existing file(s) into '{summary.output_dir}'."
    )
    return 0


def _build_kaggle_api() -> KaggleDatasetApi:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'kaggle' package is required. Install project dependencies with "
            "'pip install -e \".\"' or install the dev environment with "
            "'pip install -e \".[dev]\"'."
        ) from exc
    return KaggleApi()


def _authenticate_api(api: KaggleDatasetApi) -> None:
    try:
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(f"Kaggle authentication failed. {AUTH_GUIDANCE}") from exc


def _list_remote_csv_files(api: KaggleDatasetApi, dataset: str) -> list[str]:
    csv_files: list[str] = []
    seen: set[str] = set()
    next_page_token: str | None = None
    seen_page_tokens: set[str] = set()

    while True:
        response = _call_dataset_list_files(
            api=api,
            dataset=dataset,
            page_token=next_page_token,
            page_size=100,
        )
        candidates = _coerce_file_entries(response)

        for candidate in candidates:
            file_name = _extract_remote_file_name(candidate)
            if not file_name:
                continue
            normalized = _normalize_remote_file_name(file_name)
            if not normalized.lower().endswith(".csv"):
                continue
            if normalized not in seen:
                seen.add(normalized)
                csv_files.append(normalized)

        next_page_token = _extract_next_page_token(response)
        if not next_page_token or next_page_token in seen_page_tokens:
            break
        seen_page_tokens.add(next_page_token)

    return csv_files


def _call_dataset_list_files(
    api: KaggleDatasetApi,
    dataset: str,
    page_token: str | None,
    page_size: int,
) -> Any:
    method = api.dataset_list_files

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        signature = None

    if signature is None:
        if page_token is not None:
            return method(dataset)
        return method(dataset)

    accepted = signature.parameters
    positional_args: list[Any] = []
    named_args: dict[str, Any] = {}

    if "dataset" in accepted:
        named_args["dataset"] = dataset
    else:
        positional_args.append(dataset)

    if page_token is not None:
        for key in ("page_token", "pageToken"):
            if key in accepted:
                named_args[key] = page_token
                break

    for key in ("page_size", "pageSize"):
        if key in accepted:
            named_args[key] = page_size
            break

    return method(*positional_args, **named_args)


def _coerce_file_entries(response: Any) -> list[Any]:
    if isinstance(response, list):
        return response
    if isinstance(response, tuple):
        return list(response)
    if isinstance(response, dict):
        files = response.get("files", [])
        return list(files) if isinstance(files, (list, tuple)) else [files]
    files_attr = getattr(response, "files", None)
    if files_attr is None:
        return [response]
    if isinstance(files_attr, (list, tuple)):
        return list(files_attr)
    return [files_attr]


def _extract_next_page_token(response: Any) -> str | None:
    if isinstance(response, dict):
        for key in ("next_page_token", "nextPageToken", "nextPage"):
            value = response.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    for attr in ("next_page_token", "nextPageToken", "next_page", "nextPage"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_remote_file_name(file_entry: Any) -> str | None:
    if isinstance(file_entry, str):
        return file_entry
    if isinstance(file_entry, dict):
        for key in ("name", "ref", "path"):
            value = file_entry.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None
    for attr in ("name", "ref", "path"):
        value = getattr(file_entry, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _normalize_remote_file_name(file_name: str) -> str:
    normalized = file_name.replace("\\", "/").strip()
    posix_path = PurePosixPath(normalized)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError(f"Unsafe Kaggle file path '{file_name}'.")
    return str(posix_path)


def _download_remote_csv(
    api: KaggleDatasetApi,
    dataset: str,
    remote_file_name: str,
    destination: Path,
    force: bool,
) -> None:
    with TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        _call_dataset_download_file(
            api=api,
            dataset=dataset,
            remote_file_name=remote_file_name,
            temp_dir=temp_dir,
            force=force,
        )

        downloaded_csv = _find_downloaded_csv(temp_dir, remote_file_name)
        if downloaded_csv is not None:
            _replace_file(downloaded_csv, destination)
            return

        downloaded_zip = _find_downloaded_zip(temp_dir, remote_file_name)
        if downloaded_zip is not None:
            _extract_csv_from_zip(downloaded_zip, remote_file_name, destination)
            return

    raise FileNotFoundError(
        f"No CSV payload for '{remote_file_name}' was produced by the Kaggle "
        "download."
    )


def _call_dataset_download_file(
    api: KaggleDatasetApi,
    dataset: str,
    remote_file_name: str,
    temp_dir: Path,
    force: bool,
) -> None:
    method = api.dataset_download_file
    kwargs: dict[str, Any] = {
        "path": str(temp_dir),
        "force": force,
        "quiet": True,
        "unzip": False,
    }

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        signature = None

    if signature is None:
        method(dataset, remote_file_name, **kwargs)
        return

    accepted = signature.parameters
    positional_args: list[Any] = []
    named_args: dict[str, Any] = {}

    if "dataset" in accepted:
        named_args["dataset"] = dataset
    else:
        positional_args.append(dataset)

    if "file_name" in accepted:
        named_args["file_name"] = remote_file_name
    elif "file" in accepted:
        named_args["file"] = remote_file_name
    else:
        positional_args.append(remote_file_name)

    for key, value in kwargs.items():
        if key in accepted:
            named_args[key] = value

    method(*positional_args, **named_args)


def _find_downloaded_csv(temp_dir: Path, remote_file_name: str) -> Path | None:
    csv_candidates = [path for path in temp_dir.rglob("*.csv") if path.is_file()]
    return _select_candidate(csv_candidates, remote_file_name)


def _find_downloaded_zip(temp_dir: Path, remote_file_name: str) -> Path | None:
    zip_candidates = [path for path in temp_dir.rglob("*.zip") if path.is_file()]
    return _select_candidate(zip_candidates, remote_file_name)


def _select_candidate(candidates: list[Path], remote_file_name: str) -> Path | None:
    if not candidates:
        return None

    remote_path = PurePosixPath(remote_file_name)
    basename = remote_path.name
    normalized_remote = remote_path.as_posix().lower()

    exact_matches = [
        path
        for path in candidates
        if PurePosixPath(path.relative_to(path.anchor).as_posix().lstrip("/")).as_posix().lower()
        == normalized_remote
    ]
    if exact_matches:
        return exact_matches[0]

    basename_matches = [path for path in candidates if path.name.lower() == basename.lower()]
    if basename_matches:
        return basename_matches[0]

    if len(candidates) == 1:
        return candidates[0]

    return None


def _extract_csv_from_zip(archive_path: Path, remote_file_name: str, destination: Path) -> None:
    remote_path = PurePosixPath(remote_file_name)
    remote_name = remote_path.name.lower()
    remote_full = remote_path.as_posix().lower()

    with ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise FileNotFoundError(
                f"Archive '{archive_path.name}' did not contain any CSV files."
            )

        exact_matches = [
            name for name in members if name.replace("\\", "/").lower() == remote_full
        ]
        basename_matches = [
            name for name in members if PurePosixPath(name).name.lower() == remote_name
        ]

        if exact_matches:
            member_name = exact_matches[0]
        elif basename_matches:
            member_name = basename_matches[0]
        elif len(members) == 1:
            member_name = members[0]
        else:
            raise FileNotFoundError(
                f"Could not identify the CSV payload for '{remote_file_name}' in "
                f"'{archive_path.name}'."
            )

        if destination.exists():
            destination.unlink()

        with archive.open(member_name) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)


def _replace_file(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    shutil.move(str(source), str(destination))


if __name__ == "__main__":
    raise SystemExit(main())
