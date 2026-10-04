# 同梱している部品と使用許諾（Third-party notices）

Sashimono 本体のソースコードは MIT License です（`LICENSE`）

配る zip（`SashimonoEdit-<版>-windows-x64.zip`）には、Sashimono を動かすための他の部品を
一緒に入れてあります その中に GPL の部品（x264・x265）があるため、**zip は全体として
GPL の条件で配ります** Sashimono 本体の MIT は GPL と両立するので、本体のソースは MIT の
ままです 部品ごとの使用許諾はそれぞれの部品に従います

## zip の中の置き場

| 置き場 | 中身 |
|---|---|
| `LICENSE.txt` | Sashimono 本体の使用許諾（MIT） |
| `THIRD_PARTY_NOTICES.txt` | この一覧 |
| `licenses\GPL-2.0.txt` `GPL-3.0.txt` `LGPL-2.1.txt` `LGPL-3.0.txt` | GNU の使用許諾の全文 |
| `licenses\<包みの名前>-<版>\` | Python の包みが自分の dist-info に持っている使用許諾の写し 組み立てのたびに集める |
| `licenses\<部品>-<版>\` | wheel が写しを持っていない部品（PyAV の DLL・LuaJIT）の写し 上流から取った物をリポジトリに置いてある（下の表） |
| `licenses\Python\LICENSE.txt` | Python 本体の使用許諾（同梱の OpenSSL・libffi・bzip2 などの分を含む） |
| `_internal\OpenGL\DLLS\` | freeglut と GLE の使用許諾（PyOpenGL が DLL と一緒に置いている物） |

## 部品の一覧

使用許諾の欄は、各部品の METADATA や使用許諾ファイルに書かれている表記です

### Python の包み

表の 1 列目は配布名（`tools/build_package.py` が組み立ての記録から数えた物と突き合わせる）

| 配布名 | 版 | 使用許諾 | ソース |
|---|---|---|---|
| `PySide6_Essentials` | 6.11.2 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only（Sashimono は LGPL-3.0 で使う） | https://code.qt.io/cgit/pyside/pyside-setup.git/ |
| `PySide6_Addons` | 6.11.2 | 同上 | 同上 |
| `shiboken6` | 6.11.2 | 同上 | 同上 |
| `av`（PyAV） | 18.1.0 | BSD-3-Clause | https://github.com/PyAV-Org/PyAV |
| `numpy` | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0（同梱の OpenBLAS・LAPACK・GCC 実行時ライブラリの分は numpy の `LICENSE.txt` にある） | https://github.com/numpy/numpy |
| `lupa` | 2.8 | MIT（Lua の使用許諾を含む） | https://github.com/scoder/lupa |
| `PyOpenGL` | 3.1.10 | BSD License（分類子の表記 wheel に使用許諾の写しが無い） | https://github.com/mcfletch/pyopengl |
| `sounddevice` | 0.5.6 | MIT | https://github.com/spatialaudio/python-sounddevice |
| `cffi` | 2.1.1 | MIT-0 | https://github.com/python-cffi/cffi |
| `cryptography` | 50.0.1 | Apache-2.0 OR BSD-3-Clause | https://github.com/pyca/cryptography |
| `attrs` | 26.1.0 | MIT | https://github.com/python-attrs/attrs |
| `pip` | 26.2.1 | MIT（中に同梱の包みの分を含む） | https://github.com/pypa/pip |
| `setuptools` | 84.0.0 | MIT | https://github.com/pypa/setuptools |
| `packaging` | 26.3 | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| `trove-classifiers` | 2026.6.1.19 | Apache Software License | https://github.com/pypa/trove-classifiers |
| `typing_extensions` | 4.16.0 | PSF-2.0 | https://github.com/python/typing_extensions |
| `pyinstaller` | 6.22.3 | GPL-2.0 に配布物への例外付き（`Sashimono.exe` の起動部と実行時の差し込み） | https://github.com/pyinstaller/pyinstaller |

版は開発機の組み立てで数えた物です 組み立てる機械の包みが変わると版も変わり、
写しはそのとき入っている物から集め直します

### Python の包みの中の DLL

PyAV の wheel（18.1.0）の DLL は、https://github.com/PyAV-Org/pyav-ffmpeg のタグ `8.1.2-1` が
組み立てた物です（PyAV の `scripts/ffmpeg-8.1.json` がこのタグを指す） 版はそのタグの
`scripts/pkg.py` と、DLL に埋め込まれた版の表記から確かめました GCC 実行時ライブラリ・
libiconv・winpthreads・zlib は、組み立てに使った MSYS2 の MinGW から写された物です

写しの欄は zip の `licenses\` の下の置き場で、取ってきた元はその右の欄です

| 部品 | 版 | 置き場 | 使用許諾 | 写し | 写しの取得元 |
|---|---|---|---|---|---|
| Qt 6（`Qt6*.dll` とプラグイン Qt6Pdf・Qt6Quick・Qt6Qml・Qt6VirtualKeyboard は使わないので積まない） | 6.11.2 | `_internal\PySide6\` | LGPL-3.0（PySide6 の表記に従う） | `LGPL-3.0.txt` `GPL-3.0.txt` | GNU の全文 |
| FFmpeg（`avcodec` `avformat` `avutil` `avfilter` `avdevice` `swscale` `swresample`） | 8.1.2 | `_internal\av.libs\` | 組み込みの表記は「LGPL version 3 or later」（`--enable-version3`） | `ffmpeg-8.1.2\LICENSE.md` | https://raw.githubusercontent.com/FFmpeg/FFmpeg/n8.1.2/LICENSE.md |
| x264（`libx264-165`） | コミット b35605ace3ddf7c1a5d67a2eb553f034aef41d55 | `_internal\av.libs\` | GPL-2.0-or-later | `x264-b35605ac\COPYING` | https://code.videolan.org/videolan/x264/-/raw/b35605ace3ddf7c1a5d67a2eb553f034aef41d55/COPYING |
| x265 | 4.2 | `_internal\av.libs\` | GPL-2.0-or-later | `x265-4.2\COPYING` | https://bitbucket.org/multicoreware/x265_git/raw/4.2/COPYING |
| dav1d | 1.5.3 | `_internal\av.libs\` | BSD-2-Clause | `dav1d-1.5.3\COPYING` | https://code.videolan.org/videolan/dav1d/-/raw/1.5.3/COPYING |
| LAME（`libmp3lame`） | 3.100 | `_internal\av.libs\` | LGPL（COPYING は GNU Library GPL 2） | `lame-3.100\COPYING` `LICENSE` | https://sourceforge.net/p/lame/svn/HEAD/tree/tags/RELEASE__3_100/lame/ |
| opencore-amr（amrnb / amrwb） | 0.1.6 | `_internal\av.libs\` | Apache-2.0 | `opencore-amr-0.1.6\LICENSE` | https://sourceforge.net/p/opencore-amr/code/ci/v0.1.6/tree/LICENSE |
| Opus | 1.6.1 | `_internal\av.libs\` | BSD-3-Clause | `opus-1.6.1\COPYING` | https://raw.githubusercontent.com/xiph/opus/v1.6.1/COPYING |
| SVT-AV1 | 4.1.0 | `_internal\av.libs\` | BSD-3-Clause-Clear と AOM Patent License 1.0 | `SVT-AV1-4.1.0\LICENSE.md` `PATENTS.md` | https://gitlab.com/AOMediaCodec/SVT-AV1/-/raw/v4.1.0/LICENSE.md |
| libvpx | 1.16.0 | `_internal\av.libs\` | BSD-3-Clause と特許の許諾 | `libvpx-1.16.0\LICENSE` `PATENTS` | https://raw.githubusercontent.com/webmproject/libvpx/v1.16.0/LICENSE |
| libwebp（`libwebp` `libwebpmux` `libsharpyuv`） | 1.6.0 | `_internal\av.libs\` | BSD-3-Clause と特許の許諾 | `libwebp-1.6.0\COPYING` `PATENTS` | https://raw.githubusercontent.com/webmproject/libwebp/v1.6.0/COPYING |
| libvpl | 2.16.0 | `_internal\av.libs\` | MIT | `libvpl-2.16.0\LICENSE` | https://raw.githubusercontent.com/intel/libvpl/v2.16.0/LICENSE |
| libiconv | 1.19 | `_internal\av.libs\` | LGPL-2.1-or-later | `libiconv-1.19\COPYING.LIB` | https://git.savannah.gnu.org/cgit/libiconv.git/plain/COPYING.LIB?h=v1.19 |
| zlib | 1.3.2 | `_internal\av.libs\` | Zlib | `zlib-1.3.2\LICENSE` | https://raw.githubusercontent.com/madler/zlib/v1.3.2/LICENSE |
| GCC 実行時ライブラリ（`libgcc_s_seh` `libstdc++`） | 16.1.0（MSYS2） | `_internal\av.libs\` | GPL-3.0-or-later WITH GCC-exception-3.1 | `gcc-16.1.0\COPYING.RUNTIME` `GPL-3.0.txt` | https://gcc.gnu.org/git/?p=gcc.git;a=blob_plain;f=COPYING.RUNTIME;hb=refs/tags/releases/gcc-16.1.0 |
| winpthreads（`libwinpthread`） | DLL から版を読めない | `_internal\av.libs\` | MIT 形式（mingw-w64 の COPYING） | `winpthreads-mingw-w64-14.0.0\COPYING` | https://raw.githubusercontent.com/mingw-w64/mingw-w64/v14.0.0/mingw-w64-libraries/winpthreads/COPYING（v14.0.0 の物） |
| OpenBLAS | numpy に同梱の版 | `_internal\numpy.libs\` | BSD-3-Clause | numpy の `LICENSE.txt` | numpy の dist-info |
| Lua 5.1〜5.5 | lupa 2.8 に同梱の版 | `_internal\lupa\` | MIT | lupa の `LICENSE.txt` | lupa の dist-info |
| LuaJIT 2.0 | コミット e4c7d8b38040518d42599eef8ddb5e67aa967a9c（lupa 2.8 が取り込んだ物） | `_internal\lupa\` | MIT | `LuaJIT-2.0-e4c7d8b3\COPYRIGHT` | https://raw.githubusercontent.com/LuaJIT/LuaJIT/e4c7d8b38040518d42599eef8ddb5e67aa967a9c/COPYRIGHT |
| LuaJIT 2.1 | コミット 18b087cd2cd4ddc4a79782bf155383a689d5093d（lupa 2.8 が取り込んだ物） | `_internal\lupa\` | MIT | `LuaJIT-2.1-18b087cd\COPYRIGHT` | https://raw.githubusercontent.com/LuaJIT/LuaJIT/18b087cd2cd4ddc4a79782bf155383a689d5093d/COPYRIGHT |
| PortAudio | sounddevice に同梱の版 | `_internal\_sounddevice_data\` | MIT（`*-asio.dll` は Steinberg の ASIO SDK を含む） | 同じフォルダの `README.md` | sounddevice の wheel |
| freeglut・GLE | PyOpenGL に同梱の版 | `_internal\OpenGL\DLLS\` | 同じフォルダの `freeglut_COPYING.txt` `gle_COPYING` | 同左 | PyOpenGL の wheel |
| Python と同梱の OpenSSL・libffi など | 3.14 | `_internal\` | PSF-2.0 ほか | `Python\LICENSE.txt` | 組み立てた Python |
| Microsoft Visual C++ 再頒布可能パッケージ（`VCRUNTIME140*.dll` `MSVCP140*.dll`） | | `_internal\` ほか | Microsoft の再頒布の条件 | `Python\LICENSE.txt` の「Additional Conditions for this Windows binary build」 | 組み立てた Python |

## ソースの入手先

**GPL と LGPL の部品の対応するソースは、zip と同じ GitHub Release に添付します**
（https://github.com/kagemorikosame/sashimono-edit/releases の、同じ版の Release に添付したソースと
`sources-manifest.json`） 上流の置き場が消えても、配った版のソースを渡せるようにするためです

添付するのは次の物で、`tools/collect_sources.py` がリリースのたびに積んだのと同じ版を落とし、
`sources-SHA256SUMS.txt` と `sources-manifest.json`（取得元・版・sha256）を一緒に書きます

| 部品 | 版 | 取得元 |
|---|---|---|
| FFmpeg | 8.1.2 | https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz |
| pyav-ffmpeg（組み立ての手順と FFmpeg・LAME・libvpx への差分） | 8.1.2-1 | https://github.com/PyAV-Org/pyav-ffmpeg/archive/refs/tags/8.1.2-1.tar.gz |
| x264 | b35605ace3ddf7c1a5d67a2eb553f034aef41d55 | https://github.com/mirror/x264/archive/b35605ace3ddf7c1a5d67a2eb553f034aef41d55.tar.gz |
| x265 | 4.2 | https://bitbucket.org/multicoreware/x265_git/downloads/x265_4.2.tar.gz |
| LAME | 3.100 | https://deb.debian.org/debian/pool/main/l/lame/lame_3.100.orig.tar.gz |
| libiconv | 1.19 | https://ftp.gnu.org/pub/gnu/libiconv/libiconv-1.19.tar.gz |
| Qt（積んだ Qt のファイルが属するモジュールだけ いまは qtbase・qtsvg・qtimageformats・qttranslations） | 6.11.2 | https://download.qt.io/official_releases/qt/6.11/6.11.2/submodules/ |
| PySide6 / shiboken6 | 6.11.2 | https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-6.11.2-src/ |

組み立てに使った FFmpeg の設定は次で確かめられます（Sashimono の開発環境で）

```
.venv\Scripts\python.exe -c "import av._core as c; print(c.library_meta['libavcodec'])"
```

Sashimono 本体のソースは https://github.com/kagemorikosame/sashimono-edit にあります

## Qt を差し替える

Qt は LGPL-3.0 で使っています zip は 1 つの exe へまとめる形ではなく、フォルダの形
（PyInstaller の onedir）で組み立てているので、Qt の DLL（`_internal\PySide6\Qt6*.dll`）と
PySide6 の部品は別のファイルのまま置かれています 使う人は、互換のある版の Qt や
PySide6 に差し替えて動かせます
