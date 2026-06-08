import os
import time
import requests
import concurrent.futures
from typing import List, Tuple, Optional, Dict

CACHE_DIR = os.path.expanduser('~/.cache/codon_model')

_best_platform_cache: Optional[str] = None

PING_TARGETS = {
    'modelscope': 'https://www.modelscope.cn',
    'huggingface': 'https://huggingface.co',
    'github': 'https://github.com'
}

def ping_platform(platform: str, timeout: float = 1.5) -> Tuple[str, float]:
    '''
    Ping a platform to measure connection latency.

    Sends a HEAD request to the platform's URL and returns the latency.

    Args:
        platform (str): The platform name to ping (modelscope, huggingface, github).
        timeout (float): Maximum time in seconds to wait for a response.

    Returns:
        Tuple[str, float]: A tuple containing the platform name and its latency in seconds.
            Returns infinity if the platform is unreachable.
    '''
    url = PING_TARGETS.get(platform)
    if not url:
        return platform, float('inf')
    try:
        start = time.perf_counter()
        response = requests.head(url, timeout=timeout)
        if response.status_code < 400:
            return platform, time.perf_counter() - start
    except requests.RequestException:
        pass
    return platform, float('inf')

def select_best_platform(platforms: List[str]) -> str:
    '''
    Select the best platform based on network latency.

    Pings all given platforms concurrently and returns the one with the lowest latency.
    Results are cached for subsequent calls.

    Args:
        platforms (List[str]): List of platform names to evaluate.

    Returns:
        str: The name of the platform with the lowest latency.
    '''
    global _best_platform_cache
    if len(platforms) == 1:
        return platforms[0]

    if _best_platform_cache is not None and _best_platform_cache in platforms:
        return _best_platform_cache
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(platforms)) as executor:
        futures = {executor.submit(ping_platform, p): p for p in platforms}
        best_platform = platforms[0]
        min_latency = float('inf')
        
        for future in concurrent.futures.as_completed(futures):
            platform, latency = future.result()
            if latency < min_latency:
                min_latency = latency
                best_platform = platform
                
    _best_platform_cache = best_platform
    return best_platform

class FileLock:
    '''
    A simple file-based lock for synchronization.

    Uses atomic file creation to implement a mutex lock. This prevents multiple
    processes from downloading the same file simultaneously.

    Attributes:
        lock_file (str): Path to the lock file.
    '''

    def __init__(self, lock_file: str):
        '''
        Initialize the FileLock.

        Args:
            lock_file (str): Path to the lock file to create.
        '''
        self.lock_file = lock_file

    def __enter__(self):
        '''
        Acquire the lock.

        Blocks until the lock can be acquired.
        '''
        while True:
            try:
                fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                time.sleep(0.5)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        '''
        Release the lock.

        Removes the lock file.
        '''
        try:
            os.remove(self.lock_file)
        except OSError:
            pass

def download_file(
    url: str,
    dest_path: str,
    desc: str = 'Downloading',
    temp_path: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None
) -> None:
    '''
    Download a file from a URL with resume support and progress tracking.

    Supports resuming interrupted downloads using Range requests. Uses a file lock
    to prevent concurrent downloads of the same file.

    Args:
        url (str): The URL to download from.
        dest_path (str): The destination path to save the file.
        desc (str): Description text for the progress display.
        temp_path (Optional[str]): Temporary path for partial downloads.
        headers (Optional[Dict[str, str]]): Additional HTTP headers to include.

    Returns:
        None
    '''
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if temp_path is None:
        temp_path = dest_path + '.tmp'
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

    lock_path = dest_path + '.lock'
    
    with FileLock(lock_path):
        if os.path.exists(dest_path): return

        req_headers = headers.copy() if headers else {}
        downloaded = 0

        if os.path.exists(temp_path):
            downloaded = os.path.getsize(temp_path)
            if downloaded > 0:
                req_headers['Range'] = f'bytes={downloaded}-'

        response = requests.get(url, headers=req_headers, stream=True, timeout=15)

        if response.status_code == 206:
            mode = 'ab'
            remaining_size = int(response.headers.get('content-length', 0))
            total_size = remaining_size + downloaded
        elif response.status_code == 200:
            mode = 'wb'
            downloaded = 0
            total_size = int(response.headers.get('content-length', 0))
        elif response.status_code == 416:
            os.replace(temp_path, dest_path)
            return
        else:
            response.raise_for_status()

        chunk_size = 1024 * 64
        start_time = time.time()
        
        print(f'{desc}:')
        with open(temp_path, mode) as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    elapsed = time.time() - start_time
                    speed = downloaded / elapsed if elapsed > 0 else 0  # Bytes/s
                    speed_mb = speed / (1024 * 1024)
                    
                    if total_size > 0:
                        percent = downloaded / total_size * 100
                        remaining_bytes = total_size - downloaded
                        eta = remaining_bytes / speed if speed > 0 else 0
                        
                        bar = '#' * int(percent // 5) + '-' * (20 - int(percent // 5))
                        print(
                            f'\r[{bar}] {percent:.1f}% | '
                            f'{downloaded/(1024*1024):.1f}MB/{total_size/(1024*1024):.1f}MB | '
                            f'{speed_mb:.2f} MB/s | ETA: {eta:.0f}s', 
                            end='', flush=True
                        )
                    else:
                        print(f'\rDownloaded {downloaded/(1024*1024):.2f}MB | {speed_mb:.2f} MB/s', end='', flush=True)
        print('\n')
        os.replace(temp_path, dest_path)