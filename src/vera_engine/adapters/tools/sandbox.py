"""Sandbox utility to confine operations to a designated workspace directory."""

from pathlib import Path


def validate_path_confinement(target: str | Path, workspace: Path) -> Path:
    """Ensures that the target path resolves inside the workspace boundaries.

    Any attempt to escape the workspace directory via symbolic links or relative paths
    (e.g., '..') raises a ValueError.

    Args:
        target: The input target filepath or directory path.
        workspace: The absolute path of the confined workspace directory.

    Returns:
        The absolute, resolved Path of the target file.

    Raises:
        ValueError: If the target path resolves outside the workspace boundary.
    """
    resolved_workspace = Path(workspace).resolve()
    resolved_target = Path(target)

    # If relative, resolve against workspace
    if not resolved_target.is_absolute():
        resolved_target = (resolved_workspace / resolved_target).resolve()
    else:
        resolved_target = resolved_target.resolve()

    # Verify boundaries: resolved_workspace must be a parent or equal to resolved_target
    if (
        resolved_workspace not in resolved_target.parents
        and resolved_target != resolved_workspace
    ):
        raise ValueError(
            f"Security Violation: Path '{target}' resolves to '{resolved_target}', "
            f"escaping workspace sandbox '{resolved_workspace}'"
        )

    return resolved_target
