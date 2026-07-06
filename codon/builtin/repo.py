import os
import hashlib
from typing import Optional, List, Dict, Literal, Union, Any
from codon.builtin.download import (
    CACHE_DIR,
    select_best_platform,
    download_file
)


PLATFORM_TEMPLATES = {
    'modelscope': {
        'model': 'https://www.modelscope.cn/models/{repo}/resolve/{branch}/{file}',
        'dataset': 'https://www.modelscope.cn/datasets/{repo}/resolve/{branch}/{file}'
    },
    'huggingface': {
        'model': 'https://huggingface.co/{repo}/resolve/{branch}/{file}',
        'dataset': 'https://huggingface.co/datasets/{repo}/resolve/{branch}/{file}'
    },
    'github': {
        'model': 'https://raw.githubusercontent.com/{repo}/{branch}/{file}',
        'dataset': 'https://raw.githubusercontent.com/{repo}/{branch}/{file}'
    }
}

def verify_sha256(file_path: str, expected_hash: str) -> bool:
    '''
    Verify the SHA256 hash of a file.

    Args:
        file_path (str): Path to the file to verify.
        expected_hash (str): Expected SHA256 hash value.

    Returns:
        bool: True if the file's hash matches the expected hash, False otherwise.
    '''
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_hash


class Repo:
    '''
    Repository manager for downloading files from multiple platforms with fallback support.

    Supports downloading models and datasets from ModelScope, HuggingFace, and GitHub.
    Automatically falls back to alternative platforms if the primary platform fails.

    Attributes:
        cache_dir (str): Directory to cache downloaded files.
        token (Optional[str]): Authentication token for private repositories.
        repo_type (Literal['model', 'dataset']): Type of repository.
        configs (Dict[str, Dict[str, Any]]): Configuration for each platform.
    '''

    def __init__(
        self,
        modelscope: Optional[Union[str, Dict[str, Any]]] = None,
        huggingface: Optional[Union[str, Dict[str, Any]]] = None,
        github: Optional[Union[str, Dict[str, Any]]] = None,
        branch: Optional[str] = None,
        repo_type: Literal['model', 'dataset'] = 'model',
        cache_dir: Optional[str] = None,
        token: Optional[str] = None
    ):
        '''
        Initialize the Repo manager.

        Args:
            modelscope (Optional[Union[str, Dict[str, Any]]]): ModelScope repository config.
                Can be a string (repo name) or dict with repo, branch, files, hashes.
            huggingface (Optional[Union[str, Dict[str, Any]]]): HuggingFace repository config.
            github (Optional[Union[str, Dict[str, Any]]]): GitHub repository config.
            branch (Optional[str]): Default branch name for all platforms.
            repo_type (Literal['model', 'dataset']): Type of repository.
            cache_dir (Optional[str]): Custom cache directory.
            token (Optional[str]): Authentication token for private repositories.
        '''
        self.cache_dir = cache_dir or CACHE_DIR
        self.token = token
        self.repo_type = repo_type
        self.configs: Dict[str, Dict[str, Any]] = {}

        for platform, cfg in [('modelscope', modelscope), ('huggingface', huggingface), ('github', github)]:
            if cfg:
                if isinstance(cfg, str):
                    self.configs[platform] = {
                        'repo': cfg,
                        'branch': branch or ('master' if platform == 'modelscope' else 'main'),
                        'files': [],
                        'hashes': {}
                    }
                elif isinstance(cfg, dict):
                    self.configs[platform] = {
                        'repo': cfg['repo'],
                        'branch': cfg.get('branch', branch or ('master' if platform == 'modelscope' else 'main')),
                        'files': cfg.get('files', []),
                        'hashes': cfg.get('hashes', {})
                    }

    def _get_headers(self, platform: str) -> Dict[str, str]:
        '''
        Get authorization headers for a platform.

        Args:
            platform (str): Platform name (modelscope, huggingface, github).

        Returns:
            Dict[str, str]: Authorization headers for the platform.
        '''
        headers = {}
        if platform == 'huggingface':
            token = self.token or os.getenv("HF_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        elif platform == 'modelscope':
            token = self.token or os.getenv("MODELSCOPE_API_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        elif platform == 'github':
            token = self.token or os.getenv("GITHUB_TOKEN")
            if token:
                headers["Authorization"] = f"token {token}"
        return headers

    def download_file_with_fallback(
        self, 
        file: str, 
        start_platform: Literal['modelscope', 'huggingface', 'github']
    ) -> str:
        '''
        Download a file from a platform with automatic fallback to alternative platforms.

        Attempts to download from the specified platform first. If it fails,
        automatically tries other configured platforms until the download succeeds.

        Args:
            file (str): The file path to download from the repository.
            start_platform (Literal['modelscope', 'huggingface', 'github']): 
                The platform to start the download from.

        Returns:
            str: Local path to the downloaded file.

        Raises:
            RuntimeError: If the file cannot be downloaded from any configured platform.
        '''
        current_platform = start_platform
        attempted_platforms = {current_platform}

        while True:
            config = self.configs[current_platform]
            repo = config['repo']
            branch = config['branch']
            hashes = config.get('hashes', {})

            repo_subdir = repo.replace('/', '_')
            local_path = os.path.join(
                self.cache_dir, 
                current_platform, 
                f"{self.repo_type}s", 
                repo_subdir, 
                branch, 
                file
            )
            shared_temp_path = os.path.join(self.cache_dir, 'temp', repo_subdir, file + '.tmp')

            if os.path.exists(local_path):
                expected_hash = hashes.get(file)
                if expected_hash and not verify_sha256(local_path, expected_hash):
                    print(f"[!] Hash mismatch for local '{file}' ({current_platform}). Redownloading...")
                    try:
                        os.remove(local_path)
                    except OSError:
                        pass
                else:
                    return local_path

            templates = PLATFORM_TEMPLATES[current_platform]
            template = templates.get(self.repo_type, templates['model'])
            download_url = template.format(repo=repo, branch=branch, file=file)
            headers = self._get_headers(current_platform)

            try:
                download_file(
                    url=download_url,
                    dest_path=local_path,
                    desc=f"Retrieving '{file}' ({current_platform} {self.repo_type})",
                    temp_path=shared_temp_path,
                    headers=headers
                )
                
                expected_hash = hashes.get(file)
                if expected_hash and not verify_sha256(local_path, expected_hash):
                    raise ValueError(f"Downloaded file '{file}' failed integrity check.")
                
                return local_path

            except Exception as e:
                print(f'\n[!] Error downloading {file} from {current_platform}: {e}')
                alt_platforms = [p for p in self.configs.keys() if p not in attempted_platforms]
                
                if not alt_platforms:
                    raise RuntimeError(f"Failed to download '{file}' from all configured platforms.") from e

                current_platform = alt_platforms[0]
                attempted_platforms.add(current_platform)
                print(f'[*] Switching platform to {current_platform} for fallback...')

    def download_file(
        self, 
        file: str, 
        platform: Optional[Literal['modelscope', 'huggingface', 'github']] = None
    ) -> str:
        '''
        Download a single file from the best available platform.

        If no platform is specified, automatically selects the best platform based on latency.
        Supports fallback to alternative platforms if the primary download fails.

        Args:
            file (str): The file path to download from the repository.
            platform (Optional[Literal['modelscope', 'huggingface', 'github']]): 
                Specific platform to use. If None, selects the best platform automatically.

        Returns:
            str: Local path to the downloaded file.

        Raises:
            ValueError: If no repository sources are configured or the specified platform is not configured.
        '''
        if not self.configs:
            raise ValueError("No repository sources configured.")

        if platform is None:
            chosen_platform = select_best_platform(list(self.configs.keys()))
        else:
            chosen_platform = platform
            if chosen_platform not in self.configs:
                raise ValueError(f"Platform '{chosen_platform}' is not configured.")

        return self.download_file_with_fallback(file, start_platform=chosen_platform)

    def download_configured_files(
        self, 
        platform: Optional[Literal['modelscope', 'huggingface', 'github']] = None
    ) -> List[str]:
        '''
        Download all configured files from the best available platform.

        Automatically checks for cached files first and prefers platforms with
        all files already cached. Supports fallback to alternative platforms.

        Args:
            platform (Optional[Literal['modelscope', 'huggingface', 'github']]): 
                Specific platform to use. If None, selects the best platform automatically.

        Returns:
            List[str]: List of local paths to all downloaded files.

        Raises:
            ValueError: If no repository sources are configured or no files are defined.
        '''
        if not self.configs:
            raise ValueError("No repository sources configured.")

        if platform is None:
            cached_platforms = []
            for plat, config in self.configs.items():
                repo = config['repo']
                files = config.get('files', [])
                branch = config['branch']
                repo_subdir = repo.replace('/', '_')
                local_dir = os.path.join(self.cache_dir, plat, f"{self.repo_type}s", repo_subdir, branch)
                if files and all(os.path.exists(os.path.join(local_dir, f)) for f in files):
                    cached_platforms.append(plat)

            if cached_platforms:
                chosen_platform = select_best_platform(cached_platforms)
            else:
                chosen_platform = select_best_platform(list(self.configs.keys()))
        else:
            chosen_platform = platform

        config = self.configs[chosen_platform]
        files = config.get('files', [])
        if not files:
            raise ValueError(f"No files defined in configuration for platform '{chosen_platform}'.")

        local_paths = []
        for file in files:
            local_path = self.download_file_with_fallback(file, start_platform=chosen_platform)
            local_paths.append(local_path)
            
        return local_paths

    @staticmethod
    def download_from_url(url: str, dest_dir: Optional[str] = None) -> str:
        '''
        Download a file from a custom URL.

        Args:
            url (str): The URL to download from.
            dest_dir (Optional[str]): Destination directory. Defaults to cache/custom_downloads.

        Returns:
            str: Local path to the downloaded file.
        '''
        filename = url.split('/')[-1].split('?')[0]
        target_dir = dest_dir or os.path.join(CACHE_DIR, 'custom_downloads')
        local_path = os.path.join(target_dir, filename)

        if not os.path.exists(local_path):
            download_file(url, local_path, desc="Downloading custom URL")
        
        return local_path