# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building standalone Shortly Studio (Windows)."""

from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

PROJECT_ROOT = Path(SPECPATH).resolve().parent

datas = [
    (str(PROJECT_ROOT / "shortsforge" / "static"), "shortsforge/static"),
    (str(PROJECT_ROOT / "shortsforge" / "assets"), "shortsforge/assets"),
]
for pkg in ("faster_whisper", "mediapipe", "certifi", "yt_dlp", "edge_tts", "tzdata"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

binaries = []
for pkg in ("ctranslate2", "onnxruntime", "cv2", "av", "mediapipe"):
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass

hiddenimports = [
    "tkinter",
    "tkinter.filedialog",
    "PIL._tkinter_finder",
    "pydantic.deprecated.decorator",
]
for pkg in (
    "shortsforge",
    "yt_dlp",
    "google.genai",
    "anthropic",
    "edge_tts",
    "faster_whisper",
    "scenedetect",
):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

a = Analysis(
    [str(PROJECT_ROOT / "gui.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "_pytest"],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ShortlyStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(PROJECT_ROOT / "installer" / "shortly.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ShortlyStudio",
)
