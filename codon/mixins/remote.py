from typing import Literal, Optional, Dict, Any, TypeVar, Union
from codon.builtin.repo import Repo
from codon.mixins.base  import CodonMixin


TRemoteResource = TypeVar('TRemoteResource', bound='RemoteResourceMixin')

class RemoteResourceMixin(CodonMixin):
    '''
    Mixin for loading resources from remote repositories.

    Provides functionality to download and load models/datasets from ModelScope,
    HuggingFace, and GitHub repositories. Supports automatic platform selection
    based on network latency and fallback mechanisms.

    Attributes:
        __modelscope__ (Optional[Union[str, Dict[str, Any]]]): ModelScope repository configuration.
        __huggingface__ (Optional[Union[str, Dict[str, Any]]]): HuggingFace repository configuration.
        __github__ (Optional[Union[str, Dict[str, Any]]]): GitHub repository configuration.
        __remote_resource__ (Optional[Union[str, Dict[str, Any]]]): Generic remote resource configuration.
    '''

    __modelscope__: Optional[Union[str, Dict[str, Any]]] = None
    __huggingface__: Optional[Union[str, Dict[str, Any]]] = None
    __github__: Optional[Union[str, Dict[str, Any]]] = None
    __remote_resource__: Optional[Union[str, Dict[str, Any]]] = None

    def from_remote(
        self: TRemoteResource,
        platform: Optional[Literal['modelscope', 'huggingface', 'github']] = None, 
        url: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        repo_type: Optional[Literal['model', 'dataset']] = None
    ) -> TRemoteResource:
        '''
        Load a resource from a remote repository or custom URL.

        Args:
            platform (Optional[Literal['modelscope', 'huggingface', 'github']]): 
                Specific platform to use. If None, selects automatically based on latency.
            url (Optional[str]): Custom URL to download from. If provided, ignores repository configs.
            cache_dir (Optional[str]): Custom cache directory.
            token (Optional[str]): Authentication token for private repositories.
            repo_type (Optional[Literal['model', 'dataset']]): Type of resource to download.

        Returns:
            TRemoteResource: The instance with loaded resource.

        Raises:
            ValueError: If no repository configuration is found.
        '''
        if url:
            local_path = Repo.download_from_url(url, dest_dir=cache_dir)
            self._dispatch_load([local_path])
            return self

        modelscope_cfg = getattr(self, '__modelscope__', None) or getattr(self, '__remote_resource__', None)
        huggingface_cfg = getattr(self, '__huggingface__', None) or getattr(self, '__remote_resource__', None)
        github_cfg = getattr(self, '__github__', None)

        if not any([modelscope_cfg, huggingface_cfg, github_cfg]):
            raise ValueError(f'No configuration found in {self.__class__.__name__}.')

        config_repo_type = None
        for cfg in [modelscope_cfg, huggingface_cfg, github_cfg]:
            if isinstance(cfg, dict) and 'repo_type' in cfg:
                config_repo_type = cfg['repo_type']
                break
        
        resolved_repo_type = repo_type or config_repo_type or 'model'

        repo = Repo(
            modelscope=modelscope_cfg,
            huggingface=huggingface_cfg,
            github=github_cfg,
            repo_type=resolved_repo_type,
            cache_dir=cache_dir,
            token=token
        )

        local_paths = repo.download_configured_files(platform=platform)

        self._dispatch_load(local_paths)
        return self

    def _dispatch_load(self, local_paths: list[str]) -> None:
        '''
        Dispatch loaded files to the appropriate loader method.

        Attempts to call loader methods in the following priority:
        1. _load_remote(local_paths) - Custom remote loader
        2. load_pretrained(target_file) - HuggingFace-style loader
        3. load(target_file) - Generic loader

        Args:
            local_paths (list[str]): List of local paths to loaded files.

        Raises:
            NotImplementedError: If no loader method is found.
        '''
        if hasattr(self, '_load_remote'):
            self._load_remote(local_paths)
        elif local_paths:
            target_file = local_paths[0]
            if hasattr(self, 'load_pretrained'):
                self.load_pretrained(target_file)
            elif hasattr(self, 'load'):
                self.load(target_file)
            else:
                raise NotImplementedError(
                    f"No loader method (_load_remote, load_pretrained, or load) "
                    f"found in {self.__class__.__name__}."
                )