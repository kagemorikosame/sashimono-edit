"""更新の目録の形 配った版が読み続ける契約なので、読み方を変えたら落ちるようにする"""

from __future__ import annotations

import json
from typing import Any

import pytest

from sashimono.links import BETA_MANIFEST_URL, STABLE_MANIFEST_URL
from sashimono.update.manifest import (
    Manifest,
    ManifestError,
    PackageInfo,
    is_newer,
    manifest_to_json,
    parse_manifest,
)

SHA = "ab" * 32


def document(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema": 1,
        "version": "1.2.3",
        "minimum": "1.0.0",
        "python_abi": "cp314",
        "notes_url": "https://github.com/owner/repo/releases/tag/v1.2.3",
        "package": {
            "url": "https://github.com/owner/repo/releases/download/v1.2.3/SashimonoEdit.zip",
            "size": 258000000,
            "sha256": SHA,
        },
    }
    data.update(changes)
    return data


def encoded(data: dict[str, Any]) -> bytes:
    return json.dumps(data).encode()


class TestTheContract:
    def test_the_documented_shape_is_read(self) -> None:
        """計画書と手順書に載せた形そのもの"""
        manifest = parse_manifest(encoded(document()))
        assert manifest == Manifest(
            schema=1,
            version="1.2.3",
            minimum="1.0.0",
            python_abi="cp314",
            notes_url="https://github.com/owner/repo/releases/tag/v1.2.3",
            package=PackageInfo(
                "https://github.com/owner/repo/releases/download/v1.2.3/SashimonoEdit.zip",
                258000000,
                SHA,
            ),
        )

    def test_added_keys_are_ignored(self) -> None:
        """キーを足すのは自由 足した目録を古い版が読めないと、足せなくなる"""
        data = document(schema=2, channel="beta")
        data["package"]["signature_hint"] = "x"
        assert parse_manifest(encoded(data)).version == "1.2.3"

    def test_it_round_trips(self) -> None:
        manifest = parse_manifest(encoded(document()))
        assert parse_manifest(manifest_to_json(manifest)) == manifest

    def test_the_fixed_urls_do_not_move(self) -> None:
        """配った版はこの URL を読み続ける 変えると、配った版が新しい版を見つけられなくなる"""
        assert STABLE_MANIFEST_URL == (
            "https://github.com/kagemorikosame/sashimono-edit/releases/latest/download/update.json"
        )
        assert BETA_MANIFEST_URL == (
            "https://github.com/kagemorikosame/sashimono-edit/releases/download/beta/update.json"
        )


class TestBrokenManifests:
    @pytest.mark.parametrize(
        "changes",
        [
            {"schema": 0},
            {"schema": True},
            {"schema": "1"},
            {"version": "最新"},
            {"version": ""},
            {"minimum": None},
            {"python_abi": "py3"},
            {"notes_url": "http://github.com/x"},
            {"package": None},
        ],
    )
    def test_a_wrong_field_is_refused(self, changes: dict[str, Any]) -> None:
        with pytest.raises(ManifestError):
            parse_manifest(encoded(document(**changes)))

    @pytest.mark.parametrize(
        "package",
        [
            {"url": "http://github.com/x.zip", "size": 1, "sha256": SHA},
            {"url": "https://evil.example.com/x.zip", "size": 1, "sha256": SHA},
            {"url": "https://github.com/x.zip", "size": 0, "sha256": SHA},
            {"url": "https://github.com/x.zip", "size": True, "sha256": SHA},
            {"url": "https://github.com/x.zip", "size": 3 * 1024**3, "sha256": SHA},
            {"url": "https://github.com/x.zip", "size": 1, "sha256": "xyz"},
            {"url": "https://github.com/x.zip", "size": 1},
        ],
    )
    def test_a_wrong_package_is_refused(self, package: dict[str, Any]) -> None:
        """zip は決めたホストからしか落とさない 大きさと SHA-256 が無ければ照らせない"""
        with pytest.raises(ManifestError):
            parse_manifest(encoded(document(package=package)))

    @pytest.mark.parametrize(
        "data",
        [b"", b"[]", b"not json", b"\xff\xfe", b"{" * 100000],
        ids=["empty", "list", "text", "bytes", "huge"],
    )
    def test_garbage_is_refused(self, data: bytes) -> None:
        """大きすぎる返事・入れ子の深い JSON も、読む前に断る"""
        with pytest.raises(ManifestError):
            parse_manifest(data)


class TestComparingVersions:
    @pytest.mark.parametrize(
        ("candidate", "current", "newer"),
        [
            ("1.2.4", "1.2.3", True),
            ("1.10.0", "1.9.0", True),
            ("1.2.0", "1.2.0b1", True),
            ("1.2.0b1", "1.2.0", False),
            ("1.2.3", "1.2.3", False),
            ("0.9", "1.0", False),
            ("壊れた", "1.0", False),
        ],
    )
    def test_versions_compare_by_the_rules(self, candidate: str, current: str, newer: bool) -> None:
        """文字で比べると 1.10 が 1.9 より古くなり、ベータが正式版より新しく見える"""
        assert is_newer(candidate, current) is newer
