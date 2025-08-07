import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path
import importlib.util
import traceback

class GitInstaller:
    """
    A ComfyUI custom node that can install GitHub repositories and their requirements automatically.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "github_url": ("STRING", {
                    "default": "https://github.com/username/repo.git",
                    "multiline": False,
                    "placeholder": "Enter GitHub repository URL"
                }),
                "install_requirements": ("BOOLEAN", {"default": True}),
                "force_reinstall": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom_folder_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Optional: custom folder name"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("status_message", "success", "installed_path")
    FUNCTION = "install_repository"
    CATEGORY = "utils/git"
    
    def __init__(self):
        # Get the custom_nodes directory relative to this file
        self.custom_nodes_dir = Path(__file__).parent.parent.absolute()
        
    def get_repo_name_from_url(self, github_url):
        """Extract repository name from GitHub URL"""
        try:
            # Handle different URL formats
            if github_url.endswith('.git'):
                repo_name = github_url.split('/')[-1][:-4]  # Remove .git
            else:
                repo_name = github_url.split('/')[-1]
            return repo_name
        except:
            return None
    
    def check_git_available(self):
        """Check if git is available in the system"""
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def clone_repository(self, github_url, target_path):
        """Clone the GitHub repository"""
        try:
            cmd = ['git', 'clone', github_url, str(target_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True, f"Repository cloned successfully: {result.stdout}"
        except subprocess.CalledProcessError as e:
            return False, f"Git clone failed: {e.stderr}"
        except Exception as e:
            return False, f"Unexpected error during clone: {str(e)}"
    
    def install_requirements(self, repo_path):
        """Install requirements.txt if it exists"""
        requirements_path = repo_path / "requirements.txt"
        
        if not requirements_path.exists():
            return True, "No requirements.txt found - skipping dependency installation"
        
        try:
            # Read requirements to check if it's empty or just comments
            with open(requirements_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Check if file has actual requirements (not just comments/empty lines)
            lines = [line.strip() for line in content.split('\n') 
                    if line.strip() and not line.strip().startswith('#')]
            
            if not lines:
                return True, "Requirements.txt contains no packages - skipping installation"
            
            # Install requirements using pip
            cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            return True, f"Requirements installed successfully: {result.stdout}"
            
        except subprocess.CalledProcessError as e:
            return False, f"Failed to install requirements: {e.stderr}"
        except Exception as e:
            return False, f"Error reading/installing requirements: {str(e)}"
    
    def reload_custom_nodes(self):
        """Attempt to reload the custom nodes system"""
        try:
            # Import nodes module to trigger reload
            import nodes
            if hasattr(nodes, 'init_external_custom_nodes'):
                # This would ideally reload custom nodes, but it's complex
                # For now, we'll just notify the user
                return "Note: Restart ComfyUI to load the newly installed custom node"
            return ""
        except Exception as e:
            return f"Note: Could not auto-reload nodes: {str(e)}"
    
    def clean_github_url(self, url):
        """Clean and normalize GitHub URL to proper git clone format"""
        try:
            url = url.strip()

            # Remove common web interface suffixes
            suffixes_to_remove = ['/tree/main/', '/tree/master/', '/tree/main', '/tree/master', '/blob/main/', '/blob/master/']
            for suffix in suffixes_to_remove:
                if url.endswith(suffix):
                    url = url[:-len(suffix)]
                    break

            # Ensure .git extension for HTTPS URLs
            if url.startswith(('https://github.com/', 'http://github.com/')):
                if not url.endswith('.git'):
                    url += '.git'

            return url
        except:
            return url

    def validate_github_url(self, url):
        """Validate if the URL is a valid GitHub repository URL"""
        try:
            url = url.strip().lower()
            if not url.startswith(('https://github.com/', 'http://github.com/', 'git@github.com:')):
                return False, "URL must be a GitHub repository URL"
            if url.startswith('git@github.com:') and not url.endswith('.git'):
                return False, "SSH URLs must end with .git"
            return True, ""
        except:
            return False, "Invalid URL format"

    def install_repository(self, github_url, install_requirements=True, force_reinstall=False, custom_folder_name=""):
        """Main function to install a GitHub repository"""

        try:
            # Validate inputs
            if not github_url or not github_url.strip():
                return "Error: GitHub URL is required", False, ""

            github_url = github_url.strip()

            # Validate GitHub URL
            url_valid, url_error = self.validate_github_url(github_url)
            if not url_valid:
                return f"Error: {url_error}", False, ""

            # Check if git is available
            if not self.check_git_available():
                return "Error: Git is not available. Please install Git first.", False, ""

            # Determine repository name and target path
            if custom_folder_name and custom_folder_name.strip():
                repo_name = custom_folder_name.strip()
                # Sanitize folder name
                repo_name = "".join(c for c in repo_name if c.isalnum() or c in ('-', '_', '.'))
                if not repo_name:
                    return "Error: Invalid custom folder name", False, ""
            else:
                repo_name = self.get_repo_name_from_url(github_url)
                if not repo_name:
                    return "Error: Could not extract repository name from URL", False, ""

            target_path = self.custom_nodes_dir / repo_name

            # Check if repository already exists
            if target_path.exists():
                if not force_reinstall:
                    return f"Repository '{repo_name}' already exists. Use force_reinstall=True to overwrite.", False, str(target_path)
                else:
                    # Remove existing directory
                    try:
                        shutil.rmtree(target_path)
                        status_messages = [f"✓ Removed existing directory: {repo_name}"]
                    except Exception as e:
                        return f"Error: Could not remove existing directory: {str(e)}", False, ""
            else:
                status_messages = []

            # Clone the repository
            clone_success, clone_message = self.clone_repository(github_url, target_path)
            if not clone_success:
                return f"Clone failed: {clone_message}", False, ""

            status_messages.append(f"✓ Repository cloned to: {target_path}")

            # Install requirements if requested
            if install_requirements:
                req_success, req_message = self.install_requirements(target_path)
                if req_success:
                    status_messages.append(f"✓ {req_message}")
                else:
                    status_messages.append(f"⚠ {req_message}")

            # Add reload note
            reload_note = self.reload_custom_nodes()
            if reload_note:
                status_messages.append(reload_note)

            final_message = "\n".join(status_messages)
            return final_message, True, str(target_path)

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
            logging.error(f"GitInstaller error: {error_msg}")
            return error_msg, False, ""
