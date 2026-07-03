"""Discovery service to perform static analysis and compile project metadata."""

from pathlib import Path

from vera_engine.core.entities import ProjectMetadata
from vera_engine.core.interfaces.tool import Tool

LANGUAGE_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "React (JS)",
    ".ts": "TypeScript",
    ".tsx": "React (TS)",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".c": "C",
    ".cpp": "C++",
    ".h": "C/C++ Header",
    ".hpp": "C++ Header",
    ".cs": "C#",
    ".rb": "Ruby",
    ".php": "PHP",
    ".sh": "Shell Script",
    ".ps1": "PowerShell Script",
    ".md": "Markdown",
    ".json": "JSON Configuration",
    ".yaml": "YAML Configuration",
    ".yml": "YAML Configuration",
    ".toml": "TOML Configuration",
    ".xml": "XML Configuration",
    ".html": "HTML",
    ".css": "CSS",
}

FRAMEWORK_FILE_MAP = {
    "package.json": "Node.js NPM",
    "pyproject.toml": "Python PEP 517 / Poetry / Hatch",
    "requirements.txt": "Python Pip",
    "poetry.lock": "Poetry Package Manager",
    "Cargo.toml": "Rust Cargo",
    "go.mod": "Go Modules",
    "pom.xml": "Java Maven",
    "build.gradle": "Java Gradle",
    "tsconfig.json": "TypeScript Project",
    "next.config.js": "Next.js Framework",
    "next.config.ts": "Next.js Framework",
    "vite.config.js": "Vite Build Tool",
    "vite.config.ts": "Vite Build Tool",
    "Dockerfile": "Docker Containerization",
    "docker-compose.yml": "Docker Compose Orchestration",
    "webpack.config.js": "Webpack Bundler",
    "tailwind.config.js": "Tailwind CSS",
}

ENTRY_POINTS = {
    "main.py",
    "app.py",
    "run.py",
    "cli.py",
    "index.js",
    "index.ts",
    "server.js",
    "main.go",
    "main.rs",
    "manage.py",
}

IMPORTANT_CONFIGS = {
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "config.yaml",
    "config.json",
    "settings.json",
    "pytest.ini",
    "ruff.toml",
    ".env.example",
    "webpack.config.js",
    "tsconfig.json",
}


class DiscoveryService:
    """Orchestrates workspace structure scans using tools to compile metadata.

    This service operates prior to execution reasoning to build static codebase facts.
    """

    def __init__(
        self,
        list_dir_tool: Tool,
        search_files_tool: Tool,
        read_file_tool: Tool,
    ) -> None:
        """Initializes the discovery service with generic filesystem tools.

        Args:
            list_dir_tool: Tool to list directory directories.
            search_files_tool: Tool to search workspace files.
            read_file_tool: Tool to read file contents.
        """
        self._list_dir = list_dir_tool
        self._search_files = search_files_tool
        self._read_file = read_file_tool

    def discover(self, workspace_path: str) -> ProjectMetadata:
        """Traverses the workspace and builds a ProjectMetadata snapshot.

        Args:
            workspace_path: Absolute path to the workspace root.

        Returns:
            A populated ProjectMetadata entity.
        """
        # 1. Fetch directory tree recursively
        list_res = self._list_dir.execute({"path": ".", "recursive": True})
        tree_output = (
            list_res.output if list_res.success else "Failed to traverse workspace."
        )

        detected_languages: set[str] = set()
        detected_frameworks: set[str] = set()
        important_files: list[str] = []
        entry_points: list[str] = []
        documentation_files: list[str] = []

        total_files = 0
        total_directories = 0
        extensions_count: dict[str, int] = {}

        # 2. Parse tree layout lines
        if list_res.success and tree_output != "Directory is empty.":
            for line in tree_output.splitlines():
                if not line.strip():
                    continue

                parts = line.split(" ", 1)
                if len(parts) < 2:
                    continue

                entry_type, filepath_str = parts[0], parts[1]
                filepath = Path(filepath_str)

                if entry_type == "[DIR]":
                    total_directories += 1
                elif entry_type == "[FILE]":
                    total_files += 1

                    # File Extension Statistics & Language Detection
                    suffix = filepath.suffix.lower()
                    if suffix:
                        extensions_count[suffix] = extensions_count.get(suffix, 0) + 1
                        if suffix in LANGUAGE_MAP:
                            detected_languages.add(LANGUAGE_MAP[suffix])

                    # Detect Frameworks by signature files
                    filename = filepath.name
                    if filename in FRAMEWORK_FILE_MAP:
                        detected_frameworks.add(FRAMEWORK_FILE_MAP[filename])

                    # Detect Entry Points
                    if filename.lower() in ENTRY_POINTS:
                        entry_points.append(filepath_str)

                    # Detect Documentation Files
                    if filename.upper().startswith("README") or suffix == ".md":
                        documentation_files.append(filepath_str)

                    # Detect Important Config Files
                    if filename in IMPORTANT_CONFIGS or filename.startswith(".env"):
                        important_files.append(filepath_str)

        file_statistics = {
            "total_files": total_files,
            "total_directories": total_directories,
            "extensions": extensions_count,
        }

        return ProjectMetadata(
            root_path=workspace_path,
            directory_tree=tree_output,
            detected_languages=sorted(detected_languages),
            detected_frameworks=sorted(detected_frameworks),
            important_files=sorted(important_files),
            entry_points=sorted(entry_points),
            documentation_files=sorted(documentation_files),
            file_statistics=file_statistics,
        )
