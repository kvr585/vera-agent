"""Local filesystem tools for the VERA agent."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from vera_engine.adapters.tools.sandbox import validate_path_confinement
from vera_engine.core.interfaces.tool import Tool, ToolResult

DEFAULT_IGNORE_DIRS = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "target",
    "coverage",
    ".cache",
    ".next",
    "__pycache__",
}


def is_binary_file(filepath: Path) -> bool:
    """Detects if a file is binary by looking for null bytes in its prefix chunk."""
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(1024)
            return b"\x00" in chunk
    except Exception:
        return True  # Treat unreadable files as binary/unsafe


class ReadFileInput(BaseModel):
    """Input arguments schema for the ReadFileTool."""

    path: str = Field(
        ...,
        description="The path to the file to read, relative to the workspace.",
    )


class ReadFileTool(Tool):
    """A secure tool that reads file contents confinement within a workspace."""

    def __init__(self, workspace_dir: str = "workspace") -> None:
        """Initializes the read file tool.

        Args:
            workspace_dir: The root workspace boundary folder.
        """
        self._workspace = Path(workspace_dir).resolve()

    @property
    def name(self) -> str:
        """The tool's registered name."""
        return "read_file"

    @property
    def description(self) -> str:
        """Human-readable description of what this tool does."""
        return (
            "Reads the text contents of a file confinement "
            "within the workspace boundary."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        """Pydantic class mapping input arguments."""
        return ReadFileInput

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Executes the file reading action securely."""
        try:
            args = self.input_schema.model_validate(arguments)
            target_path = Path(args.path)

            validate_path_confinement(target_path, self._workspace)

            resolved_path = (self._workspace / target_path).resolve()
            if not resolved_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {args.path}",
                )

            if not resolved_path.is_file():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Path is a directory, not a file: {args.path}",
                )

            content = resolved_path.read_text(encoding="utf-8")
            return ToolResult(success=True, output=content)

        except ValueError as err:
            return ToolResult(
                success=False,
                output="",
                error=f"Path confinement security violation: {err}",
            )
        except Exception as err:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to read file: {err}",
            )


class ListDirectoryInput(BaseModel):
    """Input arguments schema for ListDirectoryTool."""

    path: str = Field(
        default=".",
        description="Path of directory to list, relative to workspace.",
    )
    recursive: bool = Field(
        default=False,
        description="If True, recursively list all subdirectories and files.",
    )
    max_depth: int | None = Field(
        default=None,
        description=(
            "Maximum depth to traverse when recursive (1-indexed relative to path)."
        ),
    )


class ListDirectoryTool(Tool):
    """Lists files and folders inside workspace, ignoring build dirs."""

    def __init__(
        self,
        workspace_dir: str = "workspace",
        ignore_dirs: set[str] | None = None,
    ) -> None:
        self._workspace = Path(workspace_dir).resolve()
        self._ignore_dirs = (
            ignore_dirs if ignore_dirs is not None else DEFAULT_IGNORE_DIRS
        )

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return (
            "Recursively or flatly lists files and directories in the workspace, "
            "ignoring build outputs and hidden directories."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return ListDirectoryInput

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            args = self.input_schema.model_validate(arguments)
            target_path = Path(args.path)
            validate_path_confinement(target_path, self._workspace)
            resolved_dir = (self._workspace / target_path).resolve()

            if not resolved_dir.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Directory not found: {args.path}",
                )
            if not resolved_dir.is_dir():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Path is not a directory: {args.path}",
                )

            results = []

            def walk(current_dir: Path, current_depth: int) -> None:
                if args.max_depth is not None and current_depth > args.max_depth:
                    return

                try:
                    for entry in sorted(current_dir.iterdir()):
                        if entry.name in self._ignore_dirs or (
                            entry.name.startswith(".") and entry.name != "."
                        ):
                            continue

                        rel_entry = entry.relative_to(resolved_dir)
                        entry_type = "[DIR]" if entry.is_dir() else "[FILE]"
                        results.append(f"{entry_type} {rel_entry.as_posix()}")

                        if args.recursive and entry.is_dir():
                            walk(entry, current_depth + 1)
                except PermissionError:
                    pass

            walk(resolved_dir, 1)
            output = "\n".join(results) if results else "Directory is empty."
            return ToolResult(success=True, output=output)

        except ValueError as err:
            return ToolResult(
                success=False, output="", error=f"Security violation: {err}"
            )
        except Exception as err:
            return ToolResult(
                success=False, output="", error=f"Failed to list directory: {err}"
            )


class SearchFilesInput(BaseModel):
    """Input arguments schema for SearchFilesTool."""

    query: str = Field(
        default="",
        description="Search query. Matches filename, extension, content, or glob.",
    )
    search_type: str = Field(
        default="filename",
        description="Search mode: 'filename', 'extension', 'content', or 'glob'.",
    )
    path: str = Field(
        default=".",
        description="Directory relative to workspace to perform search under.",
    )
    max_file_size_kb: int = Field(
        default=1024,
        description="Max file size in KB to inspect during content search.",
    )
    exclude_dirs: list[str] = Field(
        default_factory=list,
        description="Optional list of directory names to skip during traversal.",
    )


class SearchFilesTool(Tool):
    """Searches workspace files by filename, extension, content, or glob."""

    def __init__(
        self,
        workspace_dir: str = "workspace",
        ignore_dirs: set[str] | None = None,
    ) -> None:
        self._workspace = Path(workspace_dir).resolve()
        self._ignore_dirs = (
            ignore_dirs if ignore_dirs is not None else DEFAULT_IGNORE_DIRS
        )

    @property
    def name(self) -> str:
        return "search_files"

    @property
    def description(self) -> str:
        return (
            "Searches for files inside the workspace using filename, extension, "
            "content substring matching, or glob pattern matching."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return SearchFilesInput

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            args = self.input_schema.model_validate(arguments)
            target_path = Path(args.path)
            validate_path_confinement(target_path, self._workspace)
            resolved_dir = (self._workspace / target_path).resolve()

            if not resolved_dir.exists() or not resolved_dir.is_dir():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Invalid search directory: {args.path}",
                )

            results = []
            ignore_set = self._ignore_dirs.union(args.exclude_dirs)

            def search_walk(current_dir: Path) -> None:
                try:
                    for entry in sorted(current_dir.iterdir()):
                        if entry.name in ignore_set or (
                            entry.name.startswith(".") and entry.name != "."
                        ):
                            continue

                        if entry.is_dir():
                            search_walk(entry)
                        elif entry.is_file():
                            rel_file = entry.relative_to(resolved_dir).as_posix()

                            if args.search_type == "filename":
                                if args.query.lower() in entry.name.lower():
                                    results.append(f"[FILE] {rel_file}")

                            elif args.search_type == "extension":
                                suffix = args.query.lower()
                                if not suffix.startswith("."):
                                    suffix = f".{suffix}"
                                if entry.suffix.lower() == suffix:
                                    results.append(f"[FILE] {rel_file}")

                            elif args.search_type == "glob":
                                if entry.match(args.query) or Path(rel_file).match(
                                    args.query
                                ):
                                    results.append(f"[FILE] {rel_file}")

                            elif args.search_type == "content":
                                if entry.stat().st_size > args.max_file_size_kb * 1024:
                                    continue
                                if is_binary_file(entry):
                                    continue

                                try:
                                    content = entry.read_text(encoding="utf-8")
                                    if args.query in content:
                                        lines = content.splitlines()
                                        for idx, line in enumerate(lines, 1):
                                            if args.query in line:
                                                snip = line.strip()[:100]
                                                m = f"[MATCH] {rel_file}:{idx} - {snip}"
                                                results.append(m)
                                except Exception:
                                    pass
                except PermissionError:
                    pass

            search_walk(resolved_dir)
            output = "\n".join(results) if results else "No matches found."
            return ToolResult(success=True, output=output)

        except ValueError as err:
            return ToolResult(
                success=False, output="", error=f"Security violation: {err}"
            )
        except Exception as err:
            return ToolResult(
                success=False, output="", error=f"Failed to search files: {err}"
            )


class WriteFileInput(BaseModel):
    """Input arguments schema for WriteFileTool."""

    path: str = Field(
        ...,
        description="Path relative to the workspace where the file will be written.",
    )
    content: str = Field(
        ...,
        description="Content text to write to the target file.",
    )


class WriteFileTool(Tool):
    """Writes content to a file inside the workspace safely."""

    def __init__(self, workspace_dir: str = "workspace") -> None:
        self._workspace = Path(workspace_dir).resolve()

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Writes text content to a target file in the workspace, "
            "creating parent directories automatically."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return WriteFileInput

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            args = self.input_schema.model_validate(arguments)
            target_path = Path(args.path)
            validate_path_confinement(target_path, self._workspace)
            resolved_path = (self._workspace / target_path).resolve()

            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_text(args.content, encoding="utf-8")

            return ToolResult(
                success=True,
                output=(
                    f"Successfully wrote {len(args.content)} "
                    f"characters to '{args.path}'."
                ),
            )

        except ValueError as err:
            return ToolResult(
                success=False, output="", error=f"Security violation: {err}"
            )
        except Exception as err:
            return ToolResult(
                success=False, output="", error=f"Failed to write file: {err}"
            )
