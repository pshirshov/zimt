# Vendored wheel set for the Intel XPU PyTorch build.
#
# nixpkgs does not (as of this writing) carry a torch-xpu build, so the
# torch / torchvision / triton-xpu wheels plus the Intel oneAPI / SYCL
# runtime libraries are pulled directly from PyTorch's XPU index.
#
# Each entry has:
#   * a fixed URL (download-r2.pytorch.org for torch / triton, PyPI for the
#     intel-* runtimes — that's where pip resolves them from)
#   * a placeholder `lib.fakeHash`, replaced by running
#     ``./scripts/seed-wheel-hashes.sh`` once (writes the right
#     ``sha256-…`` strings in-place).
#
# Re-pin for newer torch / Intel oneAPI versions by editing the URLs +
# running the script again.
#
# Layout note (why all wheels are combined into one derivation):
# ----------------------------------------------------------------
# The upstream torch+xpu wheel was built with ``auditwheel repair`` and
# its libraries carry RPATHs like
#   ``$ORIGIN/../../../..``     (libtorch_xpu.so)
#   ``$ORIGIN``                  (libsycl.so.8)
# i.e. they expect every wheel in the set to be unpacked into a single
# ``<prefix>`` tree:
#   <prefix>/lib/python3.X/site-packages/torch/lib/libtorch_xpu.so
#   <prefix>/lib/libsycl.so.8           (from intel-sycl-rt's
#                                        ``.data/data/lib/`` scheme)
# A normal ``pip install`` produces exactly that layout because all the
# wheels land in the same venv. Nix's default ``buildPythonPackage`` per
# wheel breaks the assumption — each wheel gets its own store path, so
# torch's ``$ORIGIN/../../../..`` resolves into torch's own store, where
# ``libsycl.so.8`` is nowhere to be found, and ``import torch`` dies with
# ``ImportError: libsycl.so.8: cannot open shared object file``.
#
# Fix: unpack the entire wheel set into one derivation honoring the
# ``*.data/{purelib,platlib,data,scripts,headers}/`` scheme, then run
# ``autoPatchelfHook`` to fix RPATHs for system libs (libstdc++,
# libgcc_s, libc, libm, libdl, libpthread). The upstream ``$ORIGIN``
# relative RPATHs are preserved (auto-patchelf augments, doesn't
# replace), so the in-tree cross-package lookups keep working.
{ python
, fetchurl
, lib
, stdenv
, autoPatchelfHook
, unzip
, zlib
}:

# torch+xpu pulls a small set of pure-python deps via its wheel metadata
# (sympy, networkx, jinja2, filelock, typing-extensions, fsspec,
# setuptools). pip would normally install those automatically; nix
# bypasses pip's metadata so we wire them in explicitly as
# ``propagatedBuildInputs`` of the combined runtime — that's enough for
# ``python.withPackages`` to pull them into the final env. Numpy /
# pillow (torchvision's deps) are already declared in package.nix.

let
  fakeHash = lib.fakeHash;
  sitePackagesRel = python.sitePackages;

  wheels = [
    # ---- Intel oneAPI / SYCL runtime ----
    {
      name = "intel_cmplr_lib_rt";
      url = "https://files.pythonhosted.org/packages/5b/f4/c59236000ce3a470bfcae4053ad96e97c3c9febf2b38e5487046b7f4505f/intel_cmplr_lib_rt-2026.1.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-jgAqr4l0HeO0JL2tUc5g6tIcyPTe6XD8y6iO48hFaS0=";
    }
    {
      name = "intel_cmplr_lib_ur";
      url = "https://files.pythonhosted.org/packages/8e/06/da0fcd62ee4672489ede80f322eec61b48a38695b0a5072d6d1075b37197/intel_cmplr_lib_ur-2026.1.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-dLZKzoJ3sDGqNjKNyPh/f5WabKTaW2jGmSPTcoPgFmQ=";
    }
    {
      name = "intel_cmplr_lic_rt";
      url = "https://files.pythonhosted.org/packages/2a/9c/cfdfb3429b32fbd2bed7a9bcce2989efa329ca1b87b70e03bc9170650512/intel_cmplr_lic_rt-2026.1.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-7J98SVvaj10mBEIFaAjudc9MGhRZEM6Zng6JZNeM4bo=";
    }
    {
      name = "intel_opencl_rt";
      url = "https://files.pythonhosted.org/packages/24/ac/08bb51b090cc1dc3ab24567901610b33d5495c76194e6714e26a2c66390a/intel_opencl_rt-2026.1.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-sokbam80fyHSqfDocvuX4a70B7TDRxRRaZTdiE6l4Z4=";
    }
    {
      name = "intel_openmp";
      url = "https://files.pythonhosted.org/packages/72/23/60aeb428e6b1fb34fb81d4970d91ff8b5deeeeb446e1628bd78f9e3d1f8b/intel_openmp-2026.1.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-5oh/cBt9IyPtFHAIiTua8i5RGqqX4PycN+uL4kpTZrA=";
    }
    {
      name = "intel_pti";
      url = "https://files.pythonhosted.org/packages/46/d4/48737239235707852fef380005f53405740b08dc5f3700c72e5b43946452/intel_pti-1.0.1-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-TnrZ6VXS2tjkHNldIuirHkdrri5uNR8Oi0ZT/nerLgw=";
    }
    {
      name = "intel_sycl_rt";
      url = "https://files.pythonhosted.org/packages/5b/9d/b183c4bdc59921b0e15006fb301b278d998abd5dc803051478de32da3476/intel_sycl_rt-2026.1.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-yPSueHwzTWpItmUUsqwpx7tK7d4pOci6Yec3DKdI/OY=";
    }
    {
      name = "mkl";
      url = "https://files.pythonhosted.org/packages/61/da/4921e17b1f455f7fed30d5cc0964f3289eee6a6cb03cdf7d5e20c14bd025/mkl-2026.1.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-TVomRJgYqK69SyqvaykYMeaR+c90sVLfSbmc9Iu9U2A=";
    }
    {
      name = "oneccl";
      url = "https://files.pythonhosted.org/packages/1c/c0/54bf02d28010584627de21af467ee34026e231528e5141069080805076d4/oneccl-2022.1.1-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-6HetNyDo1t1F1xzwXdnWXmb1o4wJAAifFUJWnkNRLng=";
    }
    {
      name = "oneccl_devel";
      url = "https://files.pythonhosted.org/packages/d5/c1/a62b38dec8add789fae282fcdaaebbb1bf55db32922fcd2fb42f8523cdc1/oneccl_devel-2022.1.1-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-oIi6Z6NhC9tiG4IzWVPOtL2mW4EUajBqG7PGTkfUdzU=";
    }
    {
      name = "onemkl_license";
      url = "https://files.pythonhosted.org/packages/e3/ef/8437c187319e779a76f4dbb468a1863d729297d79a1b5f44b10a58c96ec2/onemkl_license-2026.1.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-Of2ClkivksngPCK6Io8HF02Cmqov48X2zXBzuOuKmAU=";
    }
    {
      name = "onemkl_sycl_blas";
      url = "https://files.pythonhosted.org/packages/d6/c5/94ce322721013c42578398846e57da4273d1f8e48ab1e49ad311a418d36e/onemkl_sycl_blas-2026.1.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-Ip3RlGK0pYjBVupzvOKf6vDpZDTXl6m644qJaU/EZJM=";
    }
    {
      name = "onemkl_sycl_dft";
      url = "https://files.pythonhosted.org/packages/71/60/80d4fa8e4e100f290572cf99856e067195fb9d22a7f462b8e0993833c526/onemkl_sycl_dft-2026.1.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-cb00+MJbaGmkQKAerdJ0MVFgG/LQEbAcESBuyB9t6As=";
    }
    {
      name = "onemkl_sycl_lapack";
      url = "https://files.pythonhosted.org/packages/0c/ee/9058fe036b82bc742adb6280574502da2a8c70364a945829579fd3dd8084/onemkl_sycl_lapack-2026.1.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-l/82JvmUY+UHNKhQNO0sp2VgG1xdOwVgaV2nrsC09ik=";
    }
    {
      name = "onemkl_sycl_rng";
      url = "https://files.pythonhosted.org/packages/f0/3c/4ee5d31e04539d1acd0a1acd3ddf722db570ac8c381d5bc2a9a197e31ca7/onemkl_sycl_rng-2026.1.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-bSTuedotf+OLmd/fPDbjny3Hlr4KtcR3o+bRwkTtXM0=";
    }
    {
      name = "onemkl_sycl_sparse";
      url = "https://files.pythonhosted.org/packages/83/af/416bc19b3488129975a9bd1dbdb8bbe9340a49a802c1e24db6728e233359/onemkl_sycl_sparse-2026.1.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-UOb+LNkBCQ1HP8VyABz6XA0ABxkDZ0XjpMlep0sy+cw=";
    }
    {
      name = "impi_rt";
      url = "https://files.pythonhosted.org/packages/0e/88/a4f4392dcf96a33e53348584f0935f800292e96d093da298fcb8088ae9dd/impi_rt-2021.18.1-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-UcqJweZOYNhqfUCn1hJDErpjqvwD+2qswUCzxmNx2BQ=";
    }
    {
      name = "dpcpp_cpp_rt";
      url = "https://files.pythonhosted.org/packages/7e/98/20fffd109174139f7dc5a3413ffca0d71e0ea0837b5d7b62b634aa44ad3d/dpcpp_cpp_rt-2026.1.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-nnBY3t8YycyOW6/1ltxs8MUygH86WTS2DpW+L2nuQBQ=";
    }
    {
      name = "tcmlib";
      url = "https://files.pythonhosted.org/packages/60/24/aa409bb20703acc70cf4d3bc620a55c789639c2995b2667fb44ae7236ec9/tcmlib-1.5.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-nXwBz/Narpv1OQtiBoDr3xCn0hHCLWSIonoClQLn0Ko=";
    }
    {
      name = "tbb";
      url = "https://files.pythonhosted.org/packages/25/0c/0266c71e3fa50a71db5ce8a1d0807863df3215c5f7b5fe7c98b257561138/tbb-2023.1.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-ZK01JBxzallUmPU0Or7I6qogPp/g29v0uG03xaOrHZw=";
    }
    {
      name = "umf";
      url = "https://files.pythonhosted.org/packages/c4/72/2e0182f4e6a727a15d0a8a99a82182a4f5bdec1a4f5767acfd2abdc72070/umf-1.1.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-VnFSxe5rjhbMVrKaipprkY3h/r+Jh39x6i6CR+85/DI=";
    }
    {
      name = "pyzes";
      url = "https://files.pythonhosted.org/packages/74/46/90e1741b3926e3b8590dfaa819439891865568fc5c90a0f5596b76ee6bff/pyzes-0.1.2-py3-none-any.whl";
      hash = "sha256-VjhoQ5kEmUBBQHGvr/+R7ofxiAdFezyw0xzV1aC32yM=";
    }

    # ---- triton-xpu ----
    {
      name = "triton_xpu";
      url = "https://download-r2.pytorch.org/whl/triton_xpu-3.8.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
      hash = "sha256-h5sCcwTmF5EanYcopMwOMVEvPMuv3NB9ulc73WDPsIU=";
    }

    # ---- torch + torchvision (cp313, +xpu) ----
    {
      name = "torch";
      url = "https://download-r2.pytorch.org/whl/xpu/torch-2.14.0%2Bxpu-cp313-cp313-manylinux_2_28_x86_64.whl";
      hash = "sha256-WvZeKJZY9wmnqvJqwTw/qRQ3NHvzkfTZu9W6jELSJ64=";
    }
    {
      name = "torchvision";
      url = "https://download-r2.pytorch.org/whl/xpu/torchvision-0.29.0%2Bxpu-cp313-cp313-manylinux_2_28_x86_64.whl";
      hash = "sha256-tomSJct+QdsJSzLXyN+KFbOeWnU4eXad6afiYuRTC3Y=";
    }
  ];

  wheelSrcs = map (w: fetchurl { inherit (w) url hash; }) wheels;

  combined = stdenv.mkDerivation {
    pname = "intel-xpu-runtime";
    version = "2026.1.0";

    srcs = wheelSrcs;

    nativeBuildInputs = [ autoPatchelfHook unzip ];
    # libstdc++ / libgcc_s come from stdenv.cc.cc.lib; zlib is a frequent
    # transitive dep of the Intel runtime libs (pti, mkl).
    buildInputs = [ stdenv.cc.cc.lib zlib ];

    # Python deps declared by the torch wheel METADATA — see header
    # comment. Without these, ``import torch._dynamo`` fails because
    # ``torch.fx.experimental.symbolic_shapes`` imports sympy.
    propagatedBuildInputs = with python.pkgs; [
      filelock
      typing-extensions
      setuptools
      sympy
      networkx
      jinja2
      fsspec
    ];

    dontUnpack = true;
    dontConfigure = true;
    dontBuild = true;
    # The wheels ship pre-stripped, pre-signed binaries — stripping would
    # only thrash the Nix hash without saving meaningful disk.
    dontStrip = true;

    # Some upstream wheels reference libs they only dlopen lazily on
    # specific hardware (e.g. NVIDIA cuda libs from torch_cpu, even
    # though we're building the xpu variant). Let auto-patchelf carry on
    # rather than failing the build for those.
    autoPatchelfIgnoreMissingDeps = true;

    installPhase = ''
      runHook preInstall

      mkdir -p "$out/${sitePackagesRel}" "$out/lib" "$out/bin" "$out/include"

      for whl in $srcs; do
        echo "unpacking $whl"
        tmp=$(mktemp -d)
        unzip -q "$whl" -d "$tmp"

        # PEP 427 wheel data scheme: ``<pkg>-<ver>.data/<scheme>/...``
        # routes to one of purelib / platlib / data / scripts / headers.
        # Everything else in the wheel goes to site-packages (purelib).
        shopt -s nullglob
        for datadir in "$tmp"/*.data; do
          [ -d "$datadir" ] || continue
          for scheme in "$datadir"/*; do
            [ -d "$scheme" ] || continue
            schemeName=$(basename "$scheme")
            case "$schemeName" in
              purelib|platlib)
                cp -af "$scheme"/. "$out/${sitePackagesRel}/"
                ;;
              data)
                cp -af "$scheme"/. "$out/"
                ;;
              scripts)
                cp -af "$scheme"/. "$out/bin/"
                ;;
              headers)
                cp -af "$scheme"/. "$out/include/"
                ;;
              *)
                echo "warning: unknown wheel scheme '$schemeName' in $whl"
                mkdir -p "$out/$schemeName"
                cp -af "$scheme"/. "$out/$schemeName/"
                ;;
            esac
          done
          rm -rf "$datadir"
        done

        # Remaining top-level entries (the actual python package +
        # ``<pkg>-<ver>.dist-info/``) go to site-packages.
        if [ -n "$(ls -A "$tmp" 2>/dev/null)" ]; then
          cp -af "$tmp"/. "$out/${sitePackagesRel}/"
        fi
        rm -rf "$tmp"
      done

      runHook postInstall
    '';
  };
in
# Single combined runtime, wrapped as a Python module so it shows up in
# ``python.withPackages``. Returned as a single-element list to keep the
# call site in package.nix unchanged (it still does ``++ xpuWheels``).
[ (python.pkgs.toPythonModule combined) ]
