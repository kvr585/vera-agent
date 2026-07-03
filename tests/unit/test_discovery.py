"""Unit tests for the VERA DiscoveryService."""

from vera_engine.core.interfaces.tool import ToolResult
from vera_engine.runtime.services.discovery import DiscoveryService


def test_discovery_service_success(mocker: any) -> None:
    """Verifies that discover correctly parses file entries and statistics."""
    mock_list = mocker.Mock()
    mock_search = mocker.Mock()
    mock_read = mocker.Mock()

    tree_lines = [
        "[FILE] main.py",
        "[FILE] README.md",
        "[DIR] src",
        "[FILE] src/index.ts",
        "[FILE] package.json",
        "[FILE] config.yaml",
        "[FILE] Dockerfile",
    ]
    mock_list.execute.return_value = ToolResult(
        success=True, output="\n".join(tree_lines)
    )

    service = DiscoveryService(
        list_dir_tool=mock_list,
        search_files_tool=mock_search,
        read_file_tool=mock_read,
    )

    metadata = service.discover(workspace_path="/test/workspace")

    assert metadata.root_path == "/test/workspace"
    assert "Python" in metadata.detected_languages
    assert "TypeScript" in metadata.detected_languages
    assert "Node.js NPM" in metadata.detected_frameworks
    assert "Docker Containerization" in metadata.detected_frameworks
    assert "main.py" in metadata.entry_points
    assert "src/index.ts" in metadata.entry_points
    assert "README.md" in metadata.documentation_files
    assert "package.json" in metadata.important_files
    assert "config.yaml" in metadata.important_files
    assert metadata.file_statistics["total_files"] == 6
    assert metadata.file_statistics["total_directories"] == 1
    assert metadata.file_statistics["extensions"][".py"] == 1


def test_discovery_service_failed_traverse(mocker: any) -> None:
    """Verifies service handles list directory failure gracefully."""
    mock_list = mocker.Mock()
    mock_search = mocker.Mock()
    mock_read = mocker.Mock()

    mock_list.execute.return_value = ToolResult(
        success=False, output="", error="Listing failed"
    )

    service = DiscoveryService(
        list_dir_tool=mock_list,
        search_files_tool=mock_search,
        read_file_tool=mock_read,
    )

    metadata = service.discover(workspace_path="/test/workspace")

    assert metadata.root_path == "/test/workspace"
    assert metadata.directory_tree == "Failed to traverse workspace."
    assert metadata.file_statistics["total_files"] == 0
