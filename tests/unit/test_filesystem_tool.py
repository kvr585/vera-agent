from pathlib import Path

from vera_engine.adapters.tools.local.filesystem import (
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)


def test_read_file_tool_success(tmp_path: Path) -> None:
    """Verifies that ReadFileTool reads existing files in the workspace successfully."""
    # Write a test file in our temp workspace
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test_file = workspace / "hello.txt"
    test_file.write_text("Hello, VERA!", encoding="utf-8")

    tool = ReadFileTool(workspace_dir=str(workspace))
    assert tool.name == "read_file"
    assert "workspace boundary" in tool.description

    # Execute read
    result = tool.execute({"path": "hello.txt"})
    assert result.success is True
    assert result.output == "Hello, VERA!"


def test_read_file_tool_not_found(tmp_path: Path) -> None:
    """Verifies that ReadFileTool returns a clean error if the file doesn't exist."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    tool = ReadFileTool(workspace_dir=str(workspace))
    result = tool.execute({"path": "missing.txt"})

    assert result.success is False
    assert "File not found" in result.error


def test_read_file_tool_directory_error(tmp_path: Path) -> None:
    """Verifies that ReadFileTool fails if the path points to a directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sub_dir = workspace / "subfolder"
    sub_dir.mkdir()

    tool = ReadFileTool(workspace_dir=str(workspace))
    result = tool.execute({"path": "subfolder"})

    assert result.success is False
    assert "directory, not a file" in result.error


def test_read_file_tool_sandbox_escape(tmp_path: Path) -> None:
    """Verifies that ReadFileTool blocks files outside the workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Secret file outside workspace
    secret_file = tmp_path / "secrets.txt"
    secret_file.write_text("Super secret details", encoding="utf-8")

    tool = ReadFileTool(workspace_dir=str(workspace))

    # Escape attempt
    result = tool.execute({"path": "../secrets.txt"})

    assert result.success is False
    assert "security violation" in result.error
    assert result.output == ""


def test_read_file_tool_exception_handling(tmp_path: Path, mocker: any) -> None:
    """Verifies that ReadFileTool wraps generic read exceptions gracefully."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test_file = workspace / "error_trigger.txt"
    test_file.write_text("Trigger", encoding="utf-8")

    tool = ReadFileTool(workspace_dir=str(workspace))

    # Force OSError on read_text
    mocker.patch.object(Path, "read_text", side_effect=OSError("Disk read failed"))

    result = tool.execute({"path": "error_trigger.txt"})

    assert result.success is False
    assert "Failed to read file" in result.error
    assert "Disk read failed" in result.error


def test_list_directory_tool(tmp_path: Path) -> None:
    """Verifies ListDirectoryTool listing and ignore filters."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("print(1)", encoding="utf-8")
    (workspace / "sub").mkdir()
    (workspace / "sub" / "b.txt").write_text("hello", encoding="utf-8")
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("", encoding="utf-8")

    tool = ListDirectoryTool(workspace_dir=str(workspace))
    assert tool.name == "list_directory"

    # Test non-recursive
    result = tool.execute({"path": "."})
    assert result.success is True
    assert "[FILE] a.py" in result.output
    assert "[DIR] sub" in result.output
    assert ".git" not in result.output

    # Test recursive
    result_rec = tool.execute({"path": ".", "recursive": True})
    assert result_rec.success is True
    assert "[FILE] sub/b.txt" in result_rec.output

    # Test depth
    result_depth = tool.execute({"path": ".", "recursive": True, "max_depth": 1})
    assert result_depth.success is True
    assert "[DIR] sub" in result_depth.output
    assert "sub/b.txt" not in result_depth.output

    # Test directory missing
    res_missing = tool.execute({"path": "invalid_dir"})
    assert res_missing.success is False
    assert "not found" in res_missing.error

    # Test directory path is a file
    res_file = tool.execute({"path": "a.py"})
    assert res_file.success is False
    assert "not a directory" in res_file.error

    # Test escape
    res_esc = tool.execute({"path": "../invalid"})
    assert res_esc.success is False
    assert "security violation" in res_esc.error.lower()


def test_search_files_tool(tmp_path: Path) -> None:
    """Verifies SearchFilesTool modes and file size / binary constraints."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "main.py").write_text("import os\n# firebase key", encoding="utf-8")
    (workspace / "README.md").write_text("# Project", encoding="utf-8")
    (workspace / "sub").mkdir()
    (workspace / "sub" / "config.yaml").write_text("api: firebase", encoding="utf-8")

    # Binary file mock write
    with open(workspace / "binary.bin", "wb") as f:
        f.write(b"\x00\x01\x02\x03\x00")

    # Too large file
    (workspace / "large.txt").write_text("firebase content " * 1000, encoding="utf-8")

    tool = SearchFilesTool(workspace_dir=str(workspace))
    assert tool.name == "search_files"

    # Filename search
    res = tool.execute({"query": "main", "search_type": "filename"})
    assert res.success is True
    assert "main.py" in res.output

    # Extension search
    res = tool.execute({"query": "py", "search_type": "extension"})
    assert res.success is True
    assert "main.py" in res.output

    # Glob search
    res = tool.execute({"query": "*.md", "search_type": "glob"})
    assert res.success is True
    assert "README.md" in res.output

    # Content search (checks small files only, ignores binary and large)
    res = tool.execute(
        {"query": "firebase", "search_type": "content", "max_file_size_kb": 2}
    )
    assert res.success is True
    assert "main.py:2" in res.output
    assert "sub/config.yaml:1" in res.output
    assert "large.txt" not in res.output
    assert "binary.bin" not in res.output

    # Test invalid directory search
    res_inv = tool.execute({"path": "non_existent"})
    assert res_inv.success is False
    assert "Invalid search directory" in res_inv.error

    # Test escape search
    res_esc = tool.execute({"path": "../escape"})
    assert res_esc.success is False
    assert "security violation" in res_esc.error.lower()


def test_write_file_tool(tmp_path: Path) -> None:
    """Verifies WriteFileTool safe writing inside workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    tool = WriteFileTool(workspace_dir=str(workspace))
    assert tool.name == "write_file"

    res = tool.execute({"path": "docs/readme.md", "content": "VERA documentation"})
    assert res.success is True
    assert "Successfully wrote" in res.output

    assert (workspace / "docs" / "readme.md").exists()
    assert (workspace / "docs" / "readme.md").read_text(
        encoding="utf-8"
    ) == "VERA documentation"

    # Test escape write
    res_esc = tool.execute({"path": "../leak.txt", "content": "leak"})
    assert res_esc.success is False
    assert "security violation" in res_esc.error.lower()
