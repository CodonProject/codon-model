import sys
import subprocess
import shutil
import ctypes
import atexit
import signal
from contextlib import ContextDecorator
from typing import Callable, Any, Optional


class ExitManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExitManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
            
        self._callbacks = []
        self._has_executed = False
        self._setup_handlers()
        self._initialized = True

    def __call__(self, func: Callable) -> Callable:
        self.register(func)
        return func

    def _setup_handlers(self):
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError: pass
            
        atexit.register(self._execute_callbacks)

    def _signal_handler(self, signum: int, frame):
        sys.exit(0)

    def _execute_callbacks(self):
        if self._has_executed: return
            
        self._has_executed = True
        if not self._callbacks: return
        
        for func, args, kwargs in reversed(self._callbacks):
            try:
                func(*args, **kwargs)
            except: pass


    def register(self, func: Optional[Callable] = None, *args: Any, **kwargs: Any):
        if func is None:
            def decorator(f: Callable) -> Callable:
                self._callbacks.append((f, args, kwargs))
                return f
            return decorator
        
        self._callbacks.append((func, args, kwargs))
        return func

exit_manager = ExitManager()

class KeepAwake(ContextDecorator):
    
    def __init__(self, screen: bool = False):
        self.screen = screen
        self._proc = None          # Linux
        self._assertion_id = None  # macOS
        self._active = False

    def __enter__(self):
        self._active = True
        platform = sys.platform
        
        try:
            if platform == 'win32':
                self._enable_windows()
            elif platform.startswith('linux'):
                self._enable_linux()
            elif platform == 'darwin':
                self._enable_macos()
            else:
                raise OSError(f'Unsupported platform: {platform}')
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f'Failed to keep system awake: {e}') from e
        
        exit_manager.register(self._cleanup)
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()

    def _cleanup(self):
        if not self._active: return
        self._active = False
        
        try:
            platform = sys.platform
            if platform == 'win32':
                self._disable_windows()
            elif platform.startswith('linux'):
                self._disable_linux()
            elif platform == 'darwin':
                self._disable_macos()
        except Exception as e:
            print(f'Warning: Failed to release keepawake lock: {e}', file=sys.stderr)

    def _enable_windows(self):
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        flags = 0x80000000 | 0x00000001
        if self.screen:
            flags |= 0x00000002  # ES_DISPLAY_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(flags)

    def _disable_windows(self):
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

    def _enable_linux(self):
        if shutil.which('systemd-inhibit'):
            what = 'sleep:idle' if self.screen else 'sleep'
            self._proc = subprocess.Popen(
                ['systemd-inhibit', f'--what={what}', 
                 '--why=Python KeepAwake', 
                 'sleep', 'infinity'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if self._proc.poll() is not None:
                raise RuntimeError('systemd-inhibit process exited immediately')
        else:
            if self.screen:
                print('Warning: No systemd-inhibit found, cannot control screen. Only sleep prevented.', file=sys.stderr)
            try:
                with open('/sys/power/wake_lock', 'w') as f: f.write('keepawake_python')
            except (OSError, PermissionError) as e:
                raise RuntimeError(f'Failed to write wake_lock (need sudo?): {e}') from e

    def _disable_linux(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        else:
            try:
                with open('/sys/power/wake_unlock', 'w') as f:
                    f.write('keepawake_python')
            except FileNotFoundError: pass

    def _enable_macos(self):
        iokit = ctypes.CDLL('/System/Library/Frameworks/IOKit.framework/IOKit', use_errno=True)
        iokit.IOPMAssertionCreateWithName.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint32)
        ]
        iokit.IOPMAssertionCreateWithName.restype = ctypes.c_int32  # kIOReturnSuccess = 0
        
        types = [b'NoIdleSleepAssertion']
        if self.screen:
            types.append(b'NoDisplaySleepAssertion')
            
        assertion_ids = []
        for t in types:
            aid = ctypes.c_uint32(0)
            ret = iokit.IOPMAssertionCreateWithName(t, 255, b'Python KeepAwake', ctypes.byref(aid))
            if ret != 0:
                for a in assertion_ids:
                    iokit.IOPMAssertionRelease(a)
                raise RuntimeError(f'macOS assertion failed with code {ret}')
            assertion_ids.append(aid.value)
        
        self._assertion_id = assertion_ids

    def _disable_macos(self):
        if self._assertion_id:
            iokit = ctypes.CDLL('/System/Library/Frameworks/IOKit.framework/IOKit')
            iokit.IOPMAssertionRelease.argtypes = [ctypes.c_uint32]
            iokit.IOPMAssertionRelease.restype = ctypes.c_int32
            for aid in self._assertion_id:
                iokit.IOPMAssertionRelease(aid)
            self._assertion_id = None 
