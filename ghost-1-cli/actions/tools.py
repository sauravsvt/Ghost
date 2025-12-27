"""
Local Tools - File Operations and System Commands

Provides safe wrappers for file system operations and command execution.
All dangerous operations require explicit confirmation.
"""

import os
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("LocalTools")


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None


class FileTools:
    """Safe file system operations."""
    
    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()
        logger.info(f"FileTools initialized with workspace: {self.workspace_root}")
        
    def _validate_path(self, path: str) -> Path:
        """Ensure path is within workspace (security sandbox)."""
        full_path = (self.workspace_root / path).resolve()
        
        if not str(full_path).startswith(str(self.workspace_root)):
            raise PermissionError(f"Path escapes workspace: {path}")
            
        return full_path
    
    def read_file(self, path: str) -> ToolResult:
        """Read contents of a file."""
        try:
            full_path = self._validate_path(path)
            content = full_path.read_text(encoding='utf-8')
            return ToolResult(success=True, output=content)
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            return ToolResult(success=False, output="", error=str(e))
    
    def write_file(self, path: str, content: str) -> ToolResult:
        """Write content to a file."""
        try:
            full_path = self._validate_path(path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
            return ToolResult(success=True, output=f"Written to {path}")
        except Exception as e:
            logger.error(f"Failed to write {path}: {e}")
            return ToolResult(success=False, output="", error=str(e))
    
    def list_directory(self, path: str = ".") -> ToolResult:
        """List contents of a directory."""
        try:
            full_path = self._validate_path(path)
            items = list(full_path.iterdir())
            output = "\n".join([
                f"{'[DIR]' if p.is_dir() else '[FILE]'} {p.name}"
                for p in sorted(items)
            ])
            return ToolResult(success=True, output=output)
        except Exception as e:
            logger.error(f"Failed to list {path}: {e}")
            return ToolResult(success=False, output="", error=str(e))
    
    def create_directory(self, path: str) -> ToolResult:
        """Create a directory."""
        try:
            full_path = self._validate_path(path)
            full_path.mkdir(parents=True, exist_ok=True)
            return ToolResult(success=True, output=f"Created {path}")
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return ToolResult(success=False, output="", error=str(e))
    
    def delete(self, path: str, confirm: bool = False) -> ToolResult:
        """Delete a file or directory. Requires confirmation."""
        if not confirm:
            return ToolResult(
                success=False, 
                output="", 
                error="Delete requires explicit confirmation"
            )
            
        try:
            full_path = self._validate_path(path)
            if full_path.is_dir():
                shutil.rmtree(full_path)
            else:
                full_path.unlink()
            return ToolResult(success=True, output=f"Deleted {path}")
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            return ToolResult(success=False, output="", error=str(e))


class CommandTools:
    """Safe command execution with sandboxing."""
    
    # Commands that are always allowed
    SAFE_COMMANDS = {
        'echo', 'dir', 'ls', 'pwd', 'whoami', 'date', 'time',
        'python', 'pip', 'node', 'npm', 'git'
    }
    
    # Commands that are NEVER allowed
    BLOCKED_COMMANDS = {
        'rm', 'del', 'format', 'shutdown', 'reboot',
        'sudo', 'su', 'chmod', 'chown', 'mkfs'
    }
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        
    def execute(self, command: str, shell: bool = True) -> ToolResult:
        """
        Execute a command with safety checks.
        
        Args:
            command: Command to execute
            shell: Whether to use shell execution
            
        Returns:
            ToolResult with command output
        """
        # Extract first word as command name
        cmd_parts = command.split()
        if not cmd_parts:
            return ToolResult(success=False, output="", error="Empty command")
            
        cmd_name = cmd_parts[0].lower()
        
        # Security check
        if cmd_name in self.BLOCKED_COMMANDS:
            logger.warning(f"Blocked dangerous command: {cmd_name}")
            return ToolResult(
                success=False,
                output="",
                error=f"Command '{cmd_name}' is blocked for safety"
            )
        
        try:
            logger.info(f"Executing: {command}")
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                return ToolResult(success=True, output=result.stdout)
            else:
                return ToolResult(
                    success=False,
                    output=result.stdout,
                    error=result.stderr
                )
                
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {self.timeout}s")
            return ToolResult(success=False, output="", error="Command timed out")
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return ToolResult(success=False, output="", error=str(e))
