#!/usr/bin/env python3
"""Offline validator for the elec-ops-inspection repo-local knowledge base."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


UTF8_BOM = b"\xef\xbb\xbf"


@dataclass
class Finding:
    level: str
    path: str
    message: str


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate .agents/knowledge without network access."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Knowledge base root. Default: inferred from this script.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return non-zero when warnings are found.",
    )
    return parser.parse_args()


def display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def add(findings: list[Finding], level: str, path: Path, message: str, root: Path) -> None:
    findings.append(Finding(level, display_path(path, root), message))


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value in ("true", "false"):
        return value == "true"
    if value.isdigit():
        return int(value)
    return value.strip("'\"")


def parse_registry_subset(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {
        "domains": {},
        "projects": {},
        "learning": {},
        "conventions": {},
    }
    section: str | None = None
    item_name: str | None = None
    current_list: tuple[str, str | None, str] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        text = raw_line.strip()

        if indent == 0:
            item_name = None
            current_list = None
            if text.endswith(":"):
                section = text[:-1]
                data.setdefault(section, {})
                continue
            key, value = text.split(":", 1)
            data[key.strip()] = parse_scalar(value)
            section = key.strip()
            continue

        if section in ("domains", "projects", "learning"):
            if indent == 2 and text.endswith(":"):
                item_name = text[:-1]
                data[section][item_name] = {}
                current_list = None
                continue
            if indent == 4 and item_name and ":" in text:
                key, value = text.split(":", 1)
                key = key.strip()
                if value.strip() == "":
                    data[section][item_name][key] = []
                    current_list = (section, item_name, key)
                else:
                    data[section][item_name][key] = parse_scalar(value)
                    current_list = None
                continue
            if indent == 6 and text.startswith("- ") and current_list:
                list_section, list_item, list_key = current_list
                data[list_section][list_item][list_key].append(parse_scalar(text[2:]))
            continue

        if section == "conventions" and indent == 2 and ":" in text:
            key, value = text.split(":", 1)
            data["conventions"][key.strip()] = parse_scalar(value)

    return data


def load_registry(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return parse_registry_subset(path)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def validate_bootstrap(
    findings: list[Finding],
    root: Path,
    registry_path: Path,
    item_label: str,
    item: dict[str, Any],
) -> None:
    kb_value = item.get("kb")
    if not kb_value:
        add(findings, "ERROR", registry_path, f"{item_label} is missing kb", root)
        return

    kb_dir = root / str(kb_value)
    if not kb_dir.is_dir():
        add(findings, "ERROR", kb_dir, f"{item_label} kb directory does not exist", root)
        return

    for rel_file in as_list(item.get("bootstrap")):
        target = kb_dir / str(rel_file)
        if not target.is_file():
            add(findings, "ERROR", target, f"{item_label} bootstrap file does not exist", root)


def validate_registry(findings: list[Finding], root: Path, registry: dict[str, Any]) -> None:
    registry_path = root / "registry.yaml"
    for section in ("domains", "projects", "learning"):
        section_items = registry.get(section, {})
        if not isinstance(section_items, dict):
            add(findings, "ERROR", registry_path, f"{section} must be a mapping", root)
            continue
        for name, item in section_items.items():
            if not isinstance(item, dict):
                add(findings, "ERROR", registry_path, f"{section}.{name} must be a mapping", root)
                continue
            validate_bootstrap(findings, root, registry_path, f"{section}.{name}", item)

    domains = registry.get("domains", {})
    projects = registry.get("projects", {})
    if isinstance(projects, dict):
        for name, item in projects.items():
            if not isinstance(item, dict):
                continue
            for domain in as_list(item.get("related_domains")):
                if domain not in domains:
                    add(
                        findings,
                        "ERROR",
                        registry_path,
                        f"projects.{name} references unknown related_domain {domain!r}",
                        root,
                    )
            workspace_value = item.get("workspace")
            if workspace_value:
                workspace = (root / str(workspace_value)).resolve()
                if not workspace.exists():
                    add(findings, "WARN", workspace, f"projects.{name}.workspace does not exist", root)


def validate_project_packages(
    findings: list[Finding], root: Path, registry: dict[str, Any]
) -> None:
    conventions = registry.get("conventions", {})
    if not isinstance(conventions, dict):
        conventions = {}
    required = [
        conventions.get("project_summary_file", "project-summary.md"),
        conventions.get("status_file", "current-status.md"),
        conventions.get("decision_file", "decisions.md"),
        conventions.get("entrypoint_file", "entrypoints.md"),
    ]

    projects = registry.get("projects", {})
    if not isinstance(projects, dict):
        return
    for name, item in projects.items():
        if not isinstance(item, dict) or not item.get("kb"):
            continue
        kb_dir = root / str(item["kb"])
        for filename in required:
            target = kb_dir / str(filename)
            if not target.is_file():
                add(findings, "ERROR", target, f"projects.{name} is missing {filename}", root)


def validate_no_utf8_bom(findings: list[Finding], root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if ".git" in path.parts or not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                prefix = handle.read(len(UTF8_BOM))
        except OSError:
            continue
        if prefix == UTF8_BOM:
            add(findings, "ERROR", path, "file starts with UTF-8 BOM", root)


def print_findings(findings: list[Finding]) -> None:
    for finding in findings:
        print(f"{finding.level}: {finding.path}: {finding.message}")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    findings: list[Finding] = []

    registry_path = root / "registry.yaml"
    if not registry_path.is_file():
        add(findings, "ERROR", registry_path, "registry.yaml does not exist", root)
        print_findings(findings)
        return 1

    registry = load_registry(registry_path)
    validate_registry(findings, root, registry)
    validate_project_packages(findings, root, registry)
    validate_no_utf8_bom(findings, root)

    print_findings(findings)
    error_count = sum(1 for finding in findings if finding.level == "ERROR")
    warn_count = sum(1 for finding in findings if finding.level == "WARN")

    if error_count:
        print(f"FAILED: {error_count} error(s), {warn_count} warning(s)")
        return 1
    if warn_count and args.fail_on_warning:
        print(f"FAILED: 0 error(s), {warn_count} warning(s)")
        return 1
    print(f"OK: offline validation passed with {warn_count} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
