# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
a = Analysis(['main.py'],
         pathex=None,
         binaries=[("/opt/anaconda3/pkgs/mkl-2021.4.0-hecd8cb5_637/lib/libmkl_intel_thread.1.dylib",".")],
         datas=[("/Users/zhihuang/Desktop/nuclei.io_pyside6/software/Artwork/icon/icon.icns","icons")],
         hiddenimports=[],
         hookspath=None,
         runtime_hooks=None,
         excludes="sklearn.externals.joblib",
         cipher=block_cipher)


pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='nuclei',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='Artwork/icon/cell-molecule-icon.png',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='nuclei',
)
app = BUNDLE(
    coll,
    name='nuclei.app',
    icon='Artwork/icon/cell-molecule-icon.png',
    bundle_identifier=None,
)





'''
splash = Splash('Artwork/toolbar/checkmark-security-lock-icon.png',
                binaries=a.binaries,
                datas=a.datas,
                text_pos=(10, 50),
                text_size=12,
                text_color='black')


app = BUNDLE(exe,
         name='nuclei.app',
         icon=None,
         bundle_identifier=None,
         version='1.0.1',
         splash,                   # <-- both, splash target
         splash.binaries           # <-- and splash binaries
         )
'''