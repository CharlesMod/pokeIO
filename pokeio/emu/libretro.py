"""Minimal libretro ctypes host — the multi-console C-core binding (#36).

The developmental-ladder foundation (see docs/specs + the ladder memory): one small
ctypes host that ``dlopen``s any libretro core .so — gambatte (GB/GBC), mgba (GBA),
parallel_n64 (N64), snes9x, … — and exposes the four primitives an RL env needs:
frame stepping, system-RAM access (for the from-boot reward taps), a video
framebuffer (for the foveal front-end), and save-state serialize/unserialize (for
Go-Explore restore / backward-robustification). "Fast native core" and "console
ladder" collapse into a single per-console ``.so`` swap.

Validated on gambatte_libretro.so (Pokémon Yellow, forced DMG): SYSTEM_RAM = the
byte-identical 8192-byte $C000-$DFFF block, serialize round-trips bit-stably, and
stepping is TAS-deterministic. This module is the low-level binding; the PokeEnv
drop-in wrapper is :mod:`pokeio.emu.retro_env`.
"""

from __future__ import annotations

import ctypes as C

import numpy as np

# --- libretro ABI constants ------------------------------------------------
RETRO_MEMORY_SAVE_RAM = 0
RETRO_MEMORY_SYSTEM_RAM = 2
RETRO_DEVICE_JOYPAD = 1
# RETRO_ENVIRONMENT commands (only the handful a headless RL host must answer)
_ENV_GET_CAN_DUPE = 3
_ENV_GET_SYSTEM_DIRECTORY = 9
_ENV_SET_PIXEL_FORMAT = 10
_ENV_GET_VARIABLE = 15
_ENV_GET_VARIABLE_UPDATE = 17
_ENV_GET_SAVE_DIRECTORY = 31
# RETRO_PIXEL_FORMAT
_FMT_0RGB1555 = 0
_FMT_XRGB8888 = 1
_FMT_RGB565 = 2

# libretro RETRO_DEVICE_ID_JOYPAD button ids
JOYPAD = {"b": 0, "y": 1, "select": 2, "start": 3, "up": 4, "down": 5,
          "left": 6, "right": 7, "a": 8, "x": 9, "l": 10, "r": 11,
          "l2": 12, "r2": 13, "l3": 14, "r3": 15}


class _retro_game_info(C.Structure):
    _fields_ = [("path", C.c_char_p), ("data", C.c_void_p),
                ("size", C.c_size_t), ("meta", C.c_char_p)]


class _retro_variable(C.Structure):
    _fields_ = [("key", C.c_char_p), ("value", C.c_char_p)]


_ENV_CB = C.CFUNCTYPE(C.c_bool, C.c_uint, C.c_void_p)
_VIDEO_CB = C.CFUNCTYPE(None, C.c_void_p, C.c_uint, C.c_uint, C.c_size_t)
_AUDIO_CB = C.CFUNCTYPE(None, C.c_int16, C.c_int16)
_AUDIOB_CB = C.CFUNCTYPE(C.c_size_t, C.c_void_p, C.c_size_t)
_POLL_CB = C.CFUNCTYPE(None)
_INPUT_CB = C.CFUNCTYPE(C.c_int16, C.c_uint, C.c_uint, C.c_uint, C.c_uint)


class LibretroCore:
    """One loaded libretro core running one game.

    ``options`` are libretro core-variable overrides (e.g. ``{"gambatte_gb_hwmode":
    "GB"}`` to force DMG so SYSTEM_RAM is the clean 8192-byte block). Buttons are
    held/released via :meth:`set_button` (name in :data:`JOYPAD`); the env layer
    drives the sticky/edge-read timing.
    """

    def __init__(self, core_path: str, rom_path: str, *, options: dict | None = None,
                 system_dir: str = "/tmp") -> None:
        self.core_path = str(core_path)
        self.rom_path = str(rom_path)
        self._options = {k: str(v).encode() for k, v in (options or {}).items()}
        self._sys_dir = system_dir.encode()
        self._pressed = 0                  # joypad bitmask (port 0)
        self._fb = None                    # (ptr, w, h, pitch) of the last frame
        self._pixfmt = _FMT_0RGB1555
        self._unknown_opts: set[str] = set()

        lib = C.CDLL(self.core_path)
        lib.retro_get_memory_size.restype = C.c_size_t
        lib.retro_get_memory_data.restype = C.c_void_p
        lib.retro_serialize_size.restype = C.c_size_t
        lib.retro_serialize.restype = C.c_bool
        lib.retro_unserialize.restype = C.c_bool
        lib.retro_load_game.restype = C.c_bool
        self.lib = lib

        # keep python callbacks alive for the core's lifetime
        self._cbs = (self._mk_env(), self._mk_video(), _AUDIO_CB(lambda l, r: None),
                     _AUDIOB_CB(lambda d, f: 0), _POLL_CB(lambda: None), self._mk_input())
        lib.retro_set_environment(self._cbs[0])
        lib.retro_set_video_refresh(self._cbs[1])
        lib.retro_set_audio_sample(self._cbs[2])
        lib.retro_set_audio_sample_batch(self._cbs[3])
        lib.retro_set_input_poll(self._cbs[4])
        lib.retro_set_input_state(self._cbs[5])
        lib.retro_init()

        self._rom = open(self.rom_path, "rb").read()
        self._rombuf = C.create_string_buffer(self._rom, len(self._rom))
        gi = _retro_game_info(path=self.rom_path.encode(),
                              data=C.cast(self._rombuf, C.c_void_p),
                              size=len(self._rom), meta=None)
        if not lib.retro_load_game(C.byref(gi)):
            raise RuntimeError(f"retro_load_game failed for {self.rom_path}")
        self._closed = False
        self.run()  # settle one frame so a framebuffer + RAM exist

    # -- callbacks ---------------------------------------------------------
    def _mk_env(self):
        def env(cmd, data):
            if cmd == _ENV_SET_PIXEL_FORMAT:
                self._pixfmt = C.cast(data, C.POINTER(C.c_int))[0]
                return True
            if cmd in (_ENV_GET_SYSTEM_DIRECTORY, _ENV_GET_SAVE_DIRECTORY):
                C.cast(data, C.POINTER(C.c_char_p))[0] = C.c_char_p(self._sys_dir)
                return True
            if cmd == _ENV_GET_CAN_DUPE:
                C.cast(data, C.POINTER(C.c_bool))[0] = True
                return True
            if cmd == _ENV_GET_VARIABLE_UPDATE:
                C.cast(data, C.POINTER(C.c_bool))[0] = False
                return True
            if cmd == _ENV_GET_VARIABLE:
                var = C.cast(data, C.POINTER(_retro_variable))[0]
                key = var.key.decode() if var.key else ""
                if key.encode() in self._options or key in self._options:
                    var.value = C.c_char_p(self._options[key] if key in self._options
                                           else self._options[key.encode()])
                    return True
                self._unknown_opts.add(key)
                return False
            return False
        return _ENV_CB(env)

    def _mk_video(self):
        def vid(data, w, h, pitch):
            self._fb = (data, int(w), int(h), int(pitch))
        return _VIDEO_CB(vid)

    def _mk_input(self):
        def inp(port, device, index, id_):
            if port != 0 or device != RETRO_DEVICE_JOYPAD:
                return 0
            return 1 if (self._pressed >> id_) & 1 else 0
        return _INPUT_CB(inp)

    # -- stepping ----------------------------------------------------------
    def run(self) -> None:
        """Advance exactly one frame (one retro_run)."""
        self.lib.retro_run()

    def set_button(self, name: str, pressed: bool) -> None:
        bit = JOYPAD[name]
        if pressed:
            self._pressed |= (1 << bit)
        else:
            self._pressed &= ~(1 << bit)

    def release_all(self) -> None:
        self._pressed = 0

    # -- memory ------------------------------------------------------------
    def system_ram(self) -> np.ndarray:
        """Zero-copy uint8 view over SYSTEM_RAM (GB: the 8192-byte $C000-$DFFF)."""
        size = self.lib.retro_get_memory_size(RETRO_MEMORY_SYSTEM_RAM)
        ptr = self.lib.retro_get_memory_data(RETRO_MEMORY_SYSTEM_RAM)
        if not ptr or not size:
            return np.zeros(0, dtype=np.uint8)
        buf = (C.c_ubyte * size).from_address(ptr)
        return np.frombuffer(buf, dtype=np.uint8)

    # -- video -------------------------------------------------------------
    def framebuffer_rgb(self) -> np.ndarray:
        """The last frame as (H, W, 3) uint8 RGB (handles XRGB8888 / RGB565 / 0RGB1555)."""
        if self._fb is None:
            return np.zeros((144, 160, 3), dtype=np.uint8)
        ptr, w, h, pitch = self._fb
        if self._pixfmt == _FMT_XRGB8888:
            row_px = pitch // 4
            raw = (C.c_uint32 * (row_px * h)).from_address(ptr)
            px = np.frombuffer(raw, dtype=np.uint32).reshape(h, row_px)[:, :w]
            r = ((px >> 16) & 0xFF).astype(np.uint8)
            g = ((px >> 8) & 0xFF).astype(np.uint8)
            b = (px & 0xFF).astype(np.uint8)
            return np.stack([r, g, b], axis=-1)
        # 16-bit formats
        row_px = pitch // 2
        raw = (C.c_uint16 * (row_px * h)).from_address(ptr)
        px = np.frombuffer(raw, dtype=np.uint16).reshape(h, row_px)[:, :w].astype(np.uint32)
        if self._pixfmt == _FMT_RGB565:
            r = ((px >> 11) & 0x1F) << 3
            g = ((px >> 5) & 0x3F) << 2
            b = (px & 0x1F) << 3
        else:  # 0RGB1555
            r = ((px >> 10) & 0x1F) << 3
            g = ((px >> 5) & 0x1F) << 3
            b = (px & 0x1F) << 3
        return np.stack([r, g, b], axis=-1).astype(np.uint8)

    # -- save state --------------------------------------------------------
    def serialize(self) -> bytes:
        size = self.lib.retro_serialize_size()
        buf = C.create_string_buffer(size)
        if not self.lib.retro_serialize(buf, size):
            raise RuntimeError("retro_serialize failed")
        return buf.raw

    def unserialize(self, data: bytes) -> None:
        buf = C.create_string_buffer(bytes(data), len(data))
        if not self.lib.retro_unserialize(buf, len(data)):
            raise RuntimeError("retro_unserialize failed")

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        try:
            self.lib.retro_unload_game()
            self.lib.retro_deinit()
        except Exception:
            pass
        self._closed = True


__all__ = ["LibretroCore", "JOYPAD", "RETRO_MEMORY_SYSTEM_RAM"]
