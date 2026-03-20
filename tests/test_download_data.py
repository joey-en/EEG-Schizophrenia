from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pytest

from eeg_schizophrenia.download_data import (
    AUTH_GUIDANCE,
    DEFAULT_DATASET,
    DownloadSummary,
    download_dataset_csvs,
    main,
)


@dataclass
class FakeFile:
    name: str


@dataclass
class FakeListResponse:
    files: list[FakeFile]
    next_page_token: str | None = None


class FakeApi:
    def __init__(
        self,
        remote_files: list[str],
        payloads: dict[str, tuple[str, str]] | None = None,
        paged_remote_files: list[list[str]] | None = None,
    ) -> None:
        self.remote_files = remote_files
        self.payloads = payloads or {}
        self.paged_remote_files = paged_remote_files
        self.download_calls: list[tuple[str, str, Path, bool]] = []
        self.list_calls: list[tuple[str, str | None, int | None]] = []
        self.authenticated = False

    def authenticate(self) -> None:
        self.authenticated = True

    def dataset_list_files(
        self, dataset: str, page_token: str | None = None, page_size: int | None = None
    ) -> FakeListResponse:
        self.list_calls.append((dataset, page_token, page_size))

        if self.paged_remote_files is None:
            return FakeListResponse([FakeFile(name=file_name) for file_name in self.remote_files])

        index = int(page_token) if page_token is not None else 0
        next_page_token = str(index + 1) if index + 1 < len(self.paged_remote_files) else None
        return FakeListResponse(
            [FakeFile(name=file_name) for file_name in self.paged_remote_files[index]],
            next_page_token=next_page_token,
        )

    def dataset_download_file(
        self,
        dataset: str,
        file_name: str,
        path: str | None = None,
        force: bool = False,
        quiet: bool = True,
        unzip: bool = False,
    ) -> None:
        target_dir = Path(path or ".")
        target_dir.mkdir(parents=True, exist_ok=True)
        self.download_calls.append((dataset, file_name, target_dir, force))

        payload_type, content = self.payloads[file_name]
        basename = PurePosixPath(file_name).name
        if payload_type == "csv":
            (target_dir / basename).write_text(content, encoding="utf-8")
            return

        if payload_type == "zip":
            archive_path = target_dir / f"{basename}.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr(file_name, content)
            return

        raise AssertionError(f"Unsupported fake payload type '{payload_type}'.")


def test_download_dataset_csvs_filters_non_csvs_and_preserves_folder_structure(tmp_path: Path) -> None:
    api = FakeApi(
        remote_files=[
            "subject_a.csv",
            "nested/subject_b.csv",
            "nested/archive.tar",
            "notes.txt",
            "subject_a.csv",
        ],
        payloads={
            "subject_a.csv": ("csv", "a,b\n1,2\n"),
            "nested/subject_b.csv": ("zip", "x,y\n3,4\n"),
        },
    )

    summary = download_dataset_csvs(output_dir=tmp_path / "raw", api=api)

    assert api.authenticated is True
    assert summary.all_csv_paths == [tmp_path / "raw" / "subject_a.csv", tmp_path / "raw" / "nested" / "subject_b.csv"]
    assert summary.downloaded_paths == summary.all_csv_paths
    assert summary.skipped_paths == []
    assert [call[1] for call in api.download_calls] == ["subject_a.csv", "nested/subject_b.csv"]
    assert not (tmp_path / "raw" / "nested" / "archive.tar").exists()
    assert (tmp_path / "raw" / "subject_a.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert (tmp_path / "raw" / "nested" / "subject_b.csv").read_text(encoding="utf-8") == "x,y\n3,4\n"


def test_download_dataset_csvs_skips_existing_files_by_default(tmp_path: Path) -> None:
    existing_file = tmp_path / "raw" / "nested" / "subject_b.csv"
    existing_file.parent.mkdir(parents=True, exist_ok=True)
    existing_file.write_text("existing\n", encoding="utf-8")

    api = FakeApi(
        remote_files=["nested/subject_b.csv"],
        payloads={"nested/subject_b.csv": ("csv", "new\n")},
    )

    summary = download_dataset_csvs(output_dir=tmp_path / "raw", api=api)

    assert summary.downloaded_paths == []
    assert summary.skipped_paths == [existing_file]
    assert api.download_calls == []
    assert existing_file.read_text(encoding="utf-8") == "existing\n"


def test_download_dataset_csvs_force_redownloads_existing_files(tmp_path: Path) -> None:
    existing_file = tmp_path / "raw" / "subject_a.csv"
    existing_file.parent.mkdir(parents=True, exist_ok=True)
    existing_file.write_text("old\n", encoding="utf-8")

    api = FakeApi(
        remote_files=["subject_a.csv"],
        payloads={"subject_a.csv": ("csv", "new\n")},
    )

    summary = download_dataset_csvs(output_dir=tmp_path / "raw", api=api, force=True)

    assert summary.downloaded_paths == [existing_file]
    assert summary.skipped_paths == []
    assert [call[1] for call in api.download_calls] == ["subject_a.csv"]
    assert existing_file.read_text(encoding="utf-8") == "new\n"


def test_download_dataset_csvs_raises_when_no_csv_files_are_listed(tmp_path: Path) -> None:
    api = FakeApi(remote_files=["archive.tar", "notes.txt"])

    with pytest.raises(ValueError, match="No CSV files"):
        download_dataset_csvs(output_dir=tmp_path / "raw", api=api)


def test_main_uses_defaults(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    recorded: dict[str, object] = {}

    def fake_download_dataset_csvs(**kwargs: object) -> DownloadSummary:
        recorded.update(kwargs)
        return DownloadSummary(
            dataset=kwargs["dataset"],
            output_dir=Path(kwargs["output_dir"]),
            all_csv_paths=[],
            downloaded_paths=[],
            skipped_paths=[],
        )

    monkeypatch.setattr("eeg_schizophrenia.download_data.download_dataset_csvs", fake_download_dataset_csvs)

    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert recorded == {
        "dataset": DEFAULT_DATASET,
        "output_dir": "data/raw",
        "force": False,
        "progress": True,
    }
    assert "downloaded 0 CSV file(s)" in captured.out


def test_main_passes_through_custom_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, object] = {}

    def fake_download_dataset_csvs(**kwargs: object) -> DownloadSummary:
        recorded.update(kwargs)
        return DownloadSummary(
            dataset=kwargs["dataset"],
            output_dir=Path(kwargs["output_dir"]),
            all_csv_paths=[],
            downloaded_paths=[],
            skipped_paths=[],
        )

    monkeypatch.setattr("eeg_schizophrenia.download_data.download_dataset_csvs", fake_download_dataset_csvs)

    exit_code = main(["--dataset", "owner/example", "--output-dir", "custom/raw", "--force"])

    assert exit_code == 0
    assert recorded == {
        "dataset": "owner/example",
        "output_dir": "custom/raw",
        "force": True,
        "progress": True,
    }


def test_download_dataset_csvs_surfaces_token_auth_guidance(tmp_path: Path) -> None:
    class FailingAuthApi(FakeApi):
        def authenticate(self) -> None:
            raise RuntimeError("bad credentials")

    api = FailingAuthApi(remote_files=["subject_a.csv"])

    with pytest.raises(RuntimeError, match="KAGGLE_API_TOKEN"):
        download_dataset_csvs(output_dir=tmp_path / "raw", api=api)

    with pytest.raises(RuntimeError, match="access_token"):
        download_dataset_csvs(output_dir=tmp_path / "raw", api=api)

    assert "KAGGLE_API_TOKEN" in AUTH_GUIDANCE


def test_download_dataset_csvs_prints_download_and_skip_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    existing_file = tmp_path / "raw" / "nested" / "subject_b.csv"
    existing_file.parent.mkdir(parents=True, exist_ok=True)
    existing_file.write_text("existing\n", encoding="utf-8")

    api = FakeApi(
        remote_files=["subject_a.csv", "nested/subject_b.csv"],
        payloads={"subject_a.csv": ("csv", "new\n"), "nested/subject_b.csv": ("csv", "ignored\n")},
    )

    download_dataset_csvs(output_dir=tmp_path / "raw", api=api, progress=True)
    captured = capsys.readouterr()

    assert "Listing CSV files in broach/button-tone-sz..." in captured.out
    assert "Downloading: subject_a.csv" in captured.out
    assert "Skipping: nested/subject_b.csv (already exists)" in captured.out


def test_download_dataset_csvs_collects_all_paginated_csv_files(tmp_path: Path) -> None:
    paged_files = [
        [f"{i}.csv/{i}.csv" for i in range(1, 21)],
        [f"{i}.csv/{i}.csv" for i in range(21, 41)],
        [f"{i}.csv/{i}.csv" for i in range(41, 82)],
    ]
    payloads = {
        file_name: ("csv", f"{file_name}\n")
        for page in paged_files
        for file_name in page
    }
    api = FakeApi(remote_files=[], payloads=payloads, paged_remote_files=paged_files)

    summary = download_dataset_csvs(output_dir=tmp_path / "raw", api=api)

    assert len(summary.all_csv_paths) == 81
    assert len(summary.downloaded_paths) == 81
    assert api.list_calls == [
        (DEFAULT_DATASET, None, 100),
        (DEFAULT_DATASET, "1", 100),
        (DEFAULT_DATASET, "2", 100),
    ]
