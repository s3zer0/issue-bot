"""
Project Analysis System for Dynamic Prompt Generation.

This module provides comprehensive project structure analysis capabilities
for identifying refactoring opportunities, code organization issues, and
generating structured reports for prompt generation contexts.
"""

import os
import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import hashlib

from loguru import logger


class IssueType(Enum):
    """Types of project issues that can be identified."""
    STRUCTURE = "structure"
    ORGANIZATION = "organization"
    DEPENDENCIES = "dependencies"
    NAMING = "naming"
    COMPLEXITY = "complexity"
    MAINTAINABILITY = "maintainability"
    PERFORMANCE = "performance"
    SECURITY = "security"


class IssueSeverity(Enum):
    """Severity levels for identified issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FileInfo:
    """Information about a single file."""
    path: str
    name: str
    extension: str
    size: int
    lines: int
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    is_test: bool = False
    is_config: bool = False
    is_main: bool = False
    complexity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "name": self.name,
            "extension": self.extension,
            "size": self.size,
            "lines": self.lines,
            "imports": self.imports,
            "exports": self.exports,
            "functions": self.functions,
            "classes": self.classes,
            "dependencies": list(self.dependencies),
            "is_test": self.is_test,
            "is_config": self.is_config,
            "is_main": self.is_main,
            "complexity_score": self.complexity_score
        }


@dataclass
class DirectoryInfo:
    """Information about a directory."""
    path: str
    name: str
    files: List[FileInfo] = field(default_factory=list)
    subdirectories: List['DirectoryInfo'] = field(default_factory=list)
    has_init: bool = False
    is_package: bool = False
    total_files: int = 0
    total_lines: int = 0
    purpose: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "name": self.name,
            "files": [f.to_dict() for f in self.files],
            "subdirectories": [d.to_dict() for d in self.subdirectories],
            "has_init": self.has_init,
            "is_package": self.is_package,
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "purpose": self.purpose
        }


@dataclass
class ProjectIssue:
    """A specific issue identified in the project."""
    issue_type: IssueType
    severity: IssueSeverity
    title: str
    description: str
    affected_files: List[str] = field(default_factory=list)
    affected_directories: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    auto_fixable: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "affected_files": self.affected_files,
            "affected_directories": self.affected_directories,
            "recommendations": self.recommendations,
            "auto_fixable": self.auto_fixable
        }


@dataclass
class ProjectAnalysis:
    """Complete project analysis results."""
    project_root: str
    structure: DirectoryInfo
    issues: List[ProjectIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    analysis_timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "project_root": self.project_root,
            "structure": self.structure.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": self.metrics,
            "recommendations": self.recommendations,
            "analysis_timestamp": self.analysis_timestamp
        }


class ProjectAnalyzer:
    """Comprehensive project analyzer for structure and issue identification."""
    
    def __init__(self, project_root: str, ignore_patterns: Optional[List[str]] = None):
        """
        Initialize the project analyzer.
        
        Args:
            project_root: Root directory of the project to analyze
            ignore_patterns: List of patterns to ignore (gitignore style)
        """
        self.project_root = Path(project_root).resolve()
        self.ignore_patterns = ignore_patterns or self._get_default_ignore_patterns()
        self.file_cache: Dict[str, FileInfo] = {}
        
        if not self.project_root.exists():
            raise ValueError(f"Project root does not exist: {project_root}")
        
        logger.info(f"Initialized ProjectAnalyzer for: {self.project_root}")
    
    def _get_default_ignore_patterns(self) -> List[str]:
        """Get default ignore patterns."""
        return [
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".git",
            ".venv",
            "venv",
            ".env",
            "node_modules",
            ".pytest_cache",
            ".coverage",
            "*.egg-info",
            "build",
            "dist",
            ".DS_Store",
            "*.log"
        ]
    
    def should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored based on patterns."""
        path_str = str(path.relative_to(self.project_root))
        
        for pattern in self.ignore_patterns:
            if pattern.startswith("*."):
                # File extension pattern
                if path_str.endswith(pattern[1:]):
                    return True
            elif pattern in path_str:
                # Substring pattern
                return True
            elif path.name == pattern:
                # Exact name match
                return True
        
        return False
    
    async def analyze_project(self) -> ProjectAnalysis:
        """
        Perform comprehensive project analysis.
        
        Returns:
            ProjectAnalysis containing all analysis results
        """
        logger.info("Starting comprehensive project analysis")
        
        # Analyze directory structure
        structure = await self._analyze_directory_structure(self.project_root)
        
        # Identify issues
        issues = await self._identify_issues(structure)
        
        # Calculate metrics
        metrics = await self._calculate_metrics(structure, issues)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(issues, metrics)
        
        analysis = ProjectAnalysis(
            project_root=str(self.project_root),
            structure=structure,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations,
            analysis_timestamp=self._get_timestamp()
        )
        
        logger.success(f"Project analysis complete: {len(issues)} issues found")
        return analysis
    
    async def _analyze_directory_structure(self, directory: Path) -> DirectoryInfo:
        """Analyze directory structure recursively."""
        dir_info = DirectoryInfo(
            path=str(directory.relative_to(self.project_root)),
            name=directory.name
        )
        
        # Check if it's a Python package
        init_file = directory / "__init__.py"
        dir_info.has_init = init_file.exists()
        dir_info.is_package = dir_info.has_init
        
        # Analyze files in this directory
        for file_path in directory.iterdir():
            if self.should_ignore(file_path):
                continue
                
            if file_path.is_file():
                file_info = await self._analyze_file(file_path)
                if file_info:
                    dir_info.files.append(file_info)
                    dir_info.total_files += 1
                    dir_info.total_lines += file_info.lines
            
            elif file_path.is_dir():
                sub_dir = await self._analyze_directory_structure(file_path)
                dir_info.subdirectories.append(sub_dir)
                dir_info.total_files += sub_dir.total_files
                dir_info.total_lines += sub_dir.total_lines
        
        # Determine directory purpose
        dir_info.purpose = self._determine_directory_purpose(dir_info)
        
        return dir_info
    
    async def _analyze_file(self, file_path: Path) -> Optional[FileInfo]:
        """Analyze a single file."""
        try:
            stat = file_path.stat()
            file_info = FileInfo(
                path=str(file_path.relative_to(self.project_root)),
                name=file_path.name,
                extension=file_path.suffix,
                size=stat.st_size,
                lines=0  # Will be updated below for specific file types
            )
            
            # Determine file type
            file_info.is_test = self._is_test_file(file_path)
            file_info.is_config = self._is_config_file(file_path)
            file_info.is_main = self._is_main_file(file_path)
            
            # Analyze Python files
            if file_path.suffix == ".py":
                await self._analyze_python_file(file_path, file_info)
            
            # Analyze other file types
            elif file_path.suffix in [".md", ".txt", ".json", ".yaml", ".yml"]:
                file_info.lines = self._count_lines(file_path)
            
            return file_info
            
        except Exception as e:
            logger.warning(f"Failed to analyze file {file_path}: {e}")
            return None
    
    async def _analyze_python_file(self, file_path: Path, file_info: FileInfo):
        """Analyze Python file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_info.lines = len(content.splitlines())
            
            # Parse AST for detailed analysis
            try:
                tree = ast.parse(content)
                await self._analyze_ast(tree, file_info)
            except SyntaxError:
                logger.warning(f"Syntax error in {file_path}, skipping AST analysis")
            
            # Analyze imports
            file_info.imports = self._extract_imports(content)
            
            # Calculate complexity
            file_info.complexity_score = self._calculate_complexity(content)
            
        except Exception as e:
            logger.warning(f"Failed to analyze Python file {file_path}: {e}")
    
    async def _analyze_ast(self, tree: ast.AST, file_info: FileInfo):
        """Analyze Python AST for functions, classes, etc."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                file_info.functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                file_info.classes.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    file_info.dependencies.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    file_info.dependencies.add(node.module.split('.')[0])
    
    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements from Python content."""
        imports = []
        
        # Find import statements
        import_pattern = r'^(?:from\s+\S+\s+)?import\s+(.+)$'
        for line in content.splitlines():
            line = line.strip()
            if line.startswith(('import ', 'from ')):
                imports.append(line)
        
        return imports
    
    def _calculate_complexity(self, content: str) -> float:
        """Calculate complexity score for file content."""
        lines = content.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Basic complexity metrics
        cyclomatic_complexity = 1  # Base complexity
        
        # Count decision points
        for line in non_empty_lines:
            stripped = line.strip()
            if any(keyword in stripped for keyword in ['if ', 'elif ', 'for ', 'while ', 'try:', 'except']):
                cyclomatic_complexity += 1
        
        # Normalize by lines of code
        loc = len(non_empty_lines)
        return cyclomatic_complexity / max(1, loc / 10)  # Per 10 lines
    
    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return len(f.readlines())
        except:
            return 0
    
    def _is_test_file(self, file_path: Path) -> bool:
        """Check if file is a test file."""
        name = file_path.name.lower()
        return (
            name.startswith('test_') or 
            name.endswith('_test.py') or
            'test' in file_path.parts
        )
    
    def _is_config_file(self, file_path: Path) -> bool:
        """Check if file is a configuration file."""
        name = file_path.name.lower()
        config_files = {
            'config.py', 'settings.py', 'configuration.py',
            '.env', '.gitignore', 'requirements.txt',
            'setup.py', 'pyproject.toml', 'pytest.ini'
        }
        return name in config_files or file_path.suffix in ['.json', '.yaml', '.yml', '.toml', '.ini']
    
    def _is_main_file(self, file_path: Path) -> bool:
        """Check if file is a main entry point."""
        name = file_path.name.lower()
        return name in ['main.py', 'app.py', 'run.py', '__main__.py', 'bot.py']
    
    def _determine_directory_purpose(self, dir_info: DirectoryInfo) -> str:
        """Determine the purpose of a directory based on its contents."""
        name = dir_info.name.lower()
        
        # Common directory purposes
        if name in ['tests', 'test']:
            return "testing"
        elif name in ['docs', 'documentation']:
            return "documentation"
        elif name in ['config', 'configuration', 'settings']:
            return "configuration"
        elif name in ['utils', 'utilities', 'helpers']:
            return "utilities"
        elif name in ['models', 'model']:
            return "data_models"
        elif name in ['views', 'controllers']:
            return "presentation"
        elif name in ['services', 'service']:
            return "business_logic"
        elif name in ['data', 'database', 'db']:
            return "data_layer"
        elif name in ['api', 'apis']:
            return "api_layer"
        elif name in ['scripts', 'tools']:
            return "scripts"
        elif dir_info.has_init:
            return "package"
        elif len(dir_info.files) > len(dir_info.subdirectories):
            return "module_collection"
        else:
            return "organization"
    
    async def _identify_issues(self, structure: DirectoryInfo) -> List[ProjectIssue]:
        """Identify various issues in the project structure."""
        issues = []
        
        # Structure issues
        issues.extend(await self._identify_structure_issues(structure))
        
        # Organization issues  
        issues.extend(await self._identify_organization_issues(structure))
        
        # Dependency issues
        issues.extend(await self._identify_dependency_issues(structure))
        
        # Complexity issues
        issues.extend(await self._identify_complexity_issues(structure))
        
        return issues
    
    async def _identify_structure_issues(self, structure: DirectoryInfo) -> List[ProjectIssue]:
        """Identify structural issues."""
        issues = []
        
        # Too many files in root directory
        root_python_files = [f for f in structure.files if f.extension == '.py']
        if len(root_python_files) > 5:
            issues.append(ProjectIssue(
                issue_type=IssueType.STRUCTURE,
                severity=IssueSeverity.MEDIUM,
                title="Too many Python files in root directory",
                description=f"Found {len(root_python_files)} Python files in root. Consider organizing into modules.",
                affected_files=[f.path for f in root_python_files],
                recommendations=[
                    "Group related files into modules",
                    "Move business logic files to appropriate packages",
                    "Keep only main entry points in root"
                ],
                auto_fixable=True
            ))
        
        # Missing __init__.py files
        missing_init_dirs = []
        await self._find_missing_init_files(structure, missing_init_dirs)
        
        if missing_init_dirs:
            issues.append(ProjectIssue(
                issue_type=IssueType.STRUCTURE,
                severity=IssueSeverity.LOW,
                title="Missing __init__.py files",
                description=f"Found {len(missing_init_dirs)} directories that could be Python packages",
                affected_directories=missing_init_dirs,
                recommendations=[
                    "Add __init__.py files to create proper Python packages",
                    "Define package-level imports and exports"
                ],
                auto_fixable=True
            ))
        
        return issues
    
    async def _find_missing_init_files(self, dir_info: DirectoryInfo, missing_dirs: List[str]):
        """Recursively find directories missing __init__.py files."""
        # Check if directory has Python files but no __init__.py
        has_python_files = any(f.extension == '.py' for f in dir_info.files)
        
        if has_python_files and not dir_info.has_init and dir_info.name != self.project_root.name:
            missing_dirs.append(dir_info.path)
        
        # Recurse into subdirectories
        for subdir in dir_info.subdirectories:
            await self._find_missing_init_files(subdir, missing_dirs)
    
    async def _identify_organization_issues(self, structure: DirectoryInfo) -> List[ProjectIssue]:
        """Identify organization issues."""
        issues = []
        
        # Mixed concerns in directories
        mixed_concern_dirs = []
        await self._find_mixed_concern_directories(structure, mixed_concern_dirs)
        
        if mixed_concern_dirs:
            issues.append(ProjectIssue(
                issue_type=IssueType.ORGANIZATION,
                severity=IssueSeverity.MEDIUM,
                title="Mixed concerns in directories",
                description="Some directories contain files with different purposes",
                affected_directories=mixed_concern_dirs,
                recommendations=[
                    "Separate concerns into different modules",
                    "Group related functionality together",
                    "Follow single responsibility principle for modules"
                ]
            ))
        
        return issues
    
    async def _find_mixed_concern_directories(self, dir_info: DirectoryInfo, mixed_dirs: List[str]):
        """Find directories with mixed concerns."""
        if not dir_info.files:
            return
        
        # Analyze file types and purposes
        purposes = set()
        for file in dir_info.files:
            if file.is_test:
                purposes.add("testing")
            elif file.is_config:
                purposes.add("configuration")
            elif file.is_main:
                purposes.add("entry_point")
            elif file.extension == ".py":
                purposes.add("code")
            else:
                purposes.add("other")
        
        # If more than 2 different purposes, it might be mixed
        if len(purposes) > 2 and "other" not in purposes:
            mixed_dirs.append(dir_info.path)
        
        # Recurse
        for subdir in dir_info.subdirectories:
            await self._find_mixed_concern_directories(subdir, mixed_dirs)
    
    async def _identify_dependency_issues(self, structure: DirectoryInfo) -> List[ProjectIssue]:
        """Identify dependency-related issues."""
        issues = []
        
        # Collect all dependencies
        all_deps = set()
        await self._collect_dependencies(structure, all_deps)
        
        # Check for common dependency issues
        if len(all_deps) > 20:
            issues.append(ProjectIssue(
                issue_type=IssueType.DEPENDENCIES,
                severity=IssueSeverity.LOW,
                title="High number of dependencies",
                description=f"Project uses {len(all_deps)} different modules/packages",
                recommendations=[
                    "Review if all dependencies are necessary",
                    "Consider consolidating similar functionality",
                    "Document dependency requirements"
                ]
            ))
        
        return issues
    
    async def _collect_dependencies(self, dir_info: DirectoryInfo, all_deps: Set[str]):
        """Collect all dependencies from directory."""
        for file in dir_info.files:
            all_deps.update(file.dependencies)
        
        for subdir in dir_info.subdirectories:
            await self._collect_dependencies(subdir, all_deps)
    
    async def _identify_complexity_issues(self, structure: DirectoryInfo) -> List[ProjectIssue]:
        """Identify complexity-related issues."""
        issues = []
        
        # Find highly complex files
        complex_files = []
        await self._find_complex_files(structure, complex_files)
        
        if complex_files:
            issues.append(ProjectIssue(
                issue_type=IssueType.COMPLEXITY,
                severity=IssueSeverity.MEDIUM,
                title="High complexity files detected",
                description=f"Found {len(complex_files)} files with high complexity scores",
                affected_files=[f[0] for f in complex_files],
                recommendations=[
                    "Consider breaking down complex functions",
                    "Extract reusable components",
                    "Apply SOLID principles"
                ]
            ))
        
        return issues
    
    async def _find_complex_files(self, dir_info: DirectoryInfo, complex_files: List[Tuple[str, float]]):
        """Find files with high complexity."""
        for file in dir_info.files:
            if file.complexity_score > 2.0:  # Threshold for high complexity
                complex_files.append((file.path, file.complexity_score))
        
        for subdir in dir_info.subdirectories:
            await self._find_complex_files(subdir, complex_files)
    
    async def _calculate_metrics(self, structure: DirectoryInfo, issues: List[ProjectIssue]) -> Dict[str, Any]:
        """Calculate project metrics."""
        metrics = {
            "total_files": structure.total_files,
            "total_lines": structure.total_lines,
            "total_issues": len(issues),
            "issue_severity_distribution": self._count_issue_severities(issues),
            "file_type_distribution": {},
            "complexity_metrics": {},
            "structure_metrics": {}
        }
        
        # File type distribution
        file_types = Counter()
        await self._count_file_types(structure, file_types)
        metrics["file_type_distribution"] = dict(file_types)
        
        # Complexity metrics
        complexities = []
        await self._collect_complexities(structure, complexities)
        
        if complexities:
            metrics["complexity_metrics"] = {
                "average_complexity": sum(complexities) / len(complexities),
                "max_complexity": max(complexities),
                "files_over_threshold": len([c for c in complexities if c > 2.0])
            }
        
        # Structure metrics
        metrics["structure_metrics"] = {
            "max_directory_depth": await self._calculate_max_depth(structure),
            "packages_count": await self._count_packages(structure),
            "modules_per_package": await self._calculate_modules_per_package(structure)
        }
        
        return metrics
    
    def _count_issue_severities(self, issues: List[ProjectIssue]) -> Dict[str, int]:
        """Count issues by severity."""
        severities = Counter(issue.severity.value for issue in issues)
        return dict(severities)
    
    async def _count_file_types(self, dir_info: DirectoryInfo, file_types: Counter):
        """Count files by type."""
        for file in dir_info.files:
            ext = file.extension or "no_extension"
            file_types[ext] += 1
        
        for subdir in dir_info.subdirectories:
            await self._count_file_types(subdir, file_types)
    
    async def _collect_complexities(self, dir_info: DirectoryInfo, complexities: List[float]):
        """Collect complexity scores."""
        for file in dir_info.files:
            if file.complexity_score > 0:
                complexities.append(file.complexity_score)
        
        for subdir in dir_info.subdirectories:
            await self._collect_complexities(subdir, complexities)
    
    async def _calculate_max_depth(self, dir_info: DirectoryInfo, current_depth: int = 0) -> int:
        """Calculate maximum directory depth."""
        if not dir_info.subdirectories:
            return current_depth
        
        max_sub_depth = 0
        for subdir in dir_info.subdirectories:
            sub_depth = await self._calculate_max_depth(subdir, current_depth + 1)
            max_sub_depth = max(max_sub_depth, sub_depth)
        
        return max_sub_depth
    
    async def _count_packages(self, dir_info: DirectoryInfo) -> int:
        """Count number of Python packages."""
        count = 1 if dir_info.is_package else 0
        
        for subdir in dir_info.subdirectories:
            count += await self._count_packages(subdir)
        
        return count
    
    async def _calculate_modules_per_package(self, dir_info: DirectoryInfo) -> float:
        """Calculate average modules per package."""
        packages = await self._count_packages(dir_info)
        python_files = 0
        await self._count_python_files(dir_info, python_files)
        
        return python_files / max(1, packages)
    
    async def _count_python_files(self, dir_info: DirectoryInfo, count_ref: List[int]):
        """Count Python files recursively."""
        if not hasattr(self, '_python_file_count'):
            self._python_file_count = 0
        
        self._python_file_count += len([f for f in dir_info.files if f.extension == '.py'])
        
        for subdir in dir_info.subdirectories:
            await self._count_python_files(subdir, count_ref)
    
    async def _generate_recommendations(self, issues: List[ProjectIssue], metrics: Dict[str, Any]) -> List[str]:
        """Generate high-level recommendations based on analysis."""
        recommendations = []
        
        # Structure recommendations
        if metrics["structure_metrics"]["max_directory_depth"] > 4:
            recommendations.append("Consider flattening deeply nested directory structure")
        
        # Complexity recommendations
        complex_metrics = metrics.get("complexity_metrics", {})
        if complex_metrics.get("files_over_threshold", 0) > 3:
            recommendations.append("Refactor high-complexity files to improve maintainability")
        
        # Issue-based recommendations
        high_severity_issues = [i for i in issues if i.severity == IssueSeverity.HIGH]
        if high_severity_issues:
            recommendations.append("Address high-severity structural issues first")
        
        # File organization recommendations
        root_files = metrics["file_type_distribution"].get(".py", 0)
        if root_files > 5:
            recommendations.append("Organize root-level Python files into logical modules")
        
        return recommendations
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def generate_refactoring_context(self, analysis: ProjectAnalysis) -> Dict[str, Any]:
        """Generate context specifically for refactoring prompts."""
        return {
            "project_structure": self._format_structure_for_prompt(analysis.structure),
            "identified_issues": [issue.title for issue in analysis.issues],
            "refactoring_goals": analysis.recommendations,
            "current_metrics": {
                "total_files": analysis.metrics["total_files"],
                "complexity_issues": analysis.metrics.get("complexity_metrics", {}).get("files_over_threshold", 0),
                "structural_issues": len([i for i in analysis.issues if i.issue_type == IssueType.STRUCTURE])
            }
        }
    
    def _format_structure_for_prompt(self, structure: DirectoryInfo, indent: int = 0) -> str:
        """Format directory structure for prompt display."""
        lines = []
        prefix = "  " * indent
        
        # Directory line
        package_indicator = " (package)" if structure.is_package else ""
        lines.append(f"{prefix}{structure.name}/{package_indicator}")
        
        # Files
        for file in sorted(structure.files, key=lambda f: f.name):
            lines.append(f"{prefix}  {file.name}")
        
        # Subdirectories
        for subdir in sorted(structure.subdirectories, key=lambda d: d.name):
            lines.append(self._format_structure_for_prompt(subdir, indent + 1))
        
        return "\n".join(lines)


# Export main classes
__all__ = [
    'IssueType',
    'IssueSeverity', 
    'FileInfo',
    'DirectoryInfo',
    'ProjectIssue',
    'ProjectAnalysis',
    'ProjectAnalyzer'
]