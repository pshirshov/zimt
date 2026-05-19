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
      url = "https://files.pythonhosted.org/packages/1b/d7/ffb7e58ac260737b5076e2738ce198468a4efb3ba1885bced038801c387a/intel_cmplr_lib_rt-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-xBPvwbLd2Kh56VGLy8N1VBuLYzlRsoJWhFcdgzrqJq0=";
    }
    {
      name = "intel_cmplr_lib_ur";
      url = "https://files.pythonhosted.org/packages/49/7e/bc668ee301350964b7be980db7d527a54a61b89a749d09bb430b4598ed3b/intel_cmplr_lib_ur-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-F1zbD7ylw+8NsXbxy9i48sNtM+t3TPHQz6BAT3SN9jQ=";
    }
    {
      name = "intel_cmplr_lic_rt";
      url = "https://files.pythonhosted.org/packages/c1/27/bbadf924bf134143895fc197701951d9d28c77bd1cfbabd5e1dfb9b90b8b/intel_cmplr_lic_rt-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-hzD7HchHQKSVYDQDUxuSRaxH0tNWOoU6exoN/jqBsrs=";
    }
    {
      name = "intel_opencl_rt";
      url = "https://files.pythonhosted.org/packages/40/6b/51459a9a6ac585ab8ca1accc58a580a23b91c5272a63ccf419c8f9d52f37/intel_opencl_rt-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-KtxqZFDKkbCgANZEPpH1ZeKTI6DxX0Yte+VB8HaoxTY=";
    }
    {
      name = "intel_openmp";
      url = "https://files.pythonhosted.org/packages/91/99/5ee1e9ae85ed3d10517d17d5a3a924fecc15b116385045a0a89eb2d5bc82/intel_openmp-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-LeXxL7KC5+J1cjU5y6M1zhgx5TP5SxxIQGJWnhvFc5k=";
    }
    {
      name = "intel_pti";
      url = "https://files.pythonhosted.org/packages/af/85/dee48118c530d9574f683f8cf3a7ad576a23f060a520335c9284ff6ba65b/intel_pti-0.16.0-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-RS5ER5pXILP+GfAPoBveopWGoaJaVXJ4hXqwo9EceFI=";
    }
    {
      name = "intel_sycl_rt";
      url = "https://files.pythonhosted.org/packages/f7/e4/047a0b42f8240a9c70f3ac120479aa1996f99cfe9fd97afb025d6bbd89a8/intel_sycl_rt-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-OxnZ9ktWgEp7xYqR/pXarPELyQa5TOPFtkFhUzZ9+XI=";
    }
    {
      name = "mkl";
      url = "https://files.pythonhosted.org/packages/b3/ee/76755ca0ec9626835e0d024c369b968f24eadce2106a7884404720670623/mkl-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-2zHln6No3U+kW0lDUfS34OYgSwjX2yeDYRjE4TcOsBE=";
    }
    {
      name = "oneccl";
      url = "https://files.pythonhosted.org/packages/73/9b/2932b6b128924ba96712ea1c807c1618b9963518eb8ec80cf834ffc3c684/oneccl-2021.17.2-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-p3wsVu66HHg3mlMKK6K70zQH9kQLREdvgyktFpQQA3M=";
    }
    {
      name = "oneccl_devel";
      url = "https://files.pythonhosted.org/packages/4a/a1/4fad3108825e2d65d812ba69a9fd3664181cfe8860e49110c92431d1629f/oneccl_devel-2021.17.2-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-bNqvEvBPqOt5NPOyPCAkEP7OeuBd/TuMxCGXAXJvGZk=";
    }
    {
      name = "onemkl_license";
      url = "https://files.pythonhosted.org/packages/3e/1d/7acbedb07bf4c71cc499527c25a3ef60bf83ed41b8918e986ed7a4573bd4/onemkl_license-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-tiY6Ab8pwE6mg+qxcS62t0wri3InIvRv2QeE7Yd4wJM=";
    }
    {
      name = "onemkl_sycl_blas";
      url = "https://files.pythonhosted.org/packages/51/17/497d29cd13029f4835383d95e644d1602dafa9b7887d298ba9ea77734dce/onemkl_sycl_blas-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-uVbBTbRJkFec83brcrpgELDIOuJ/wTL2Z1KxszfZ5yg=";
    }
    {
      name = "onemkl_sycl_dft";
      url = "https://files.pythonhosted.org/packages/41/ae/46fe3ca4fcf715cfef35b239abe705d32f355c3be4b9e94aca782a4720ae/onemkl_sycl_dft-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-OfQ6M00Fq1mJsizrRiQ+rM5FcdAWyfZ5Ixef9OQnuIo=";
    }
    {
      name = "onemkl_sycl_lapack";
      url = "https://files.pythonhosted.org/packages/24/f6/f4e38bb1a81fbda3afa2d79aa95b5a5d73c838977c1f86fbf73e7dafe676/onemkl_sycl_lapack-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-Ng/GNrwIMtWi7i08T0QBRexD2UwXCjUdWIqa96hISc4=";
    }
    {
      name = "onemkl_sycl_rng";
      url = "https://files.pythonhosted.org/packages/60/f9/1f3b6ce37848c721462b4f30de08482dc27d6ffbcbba621b61ef53e28d3c/onemkl_sycl_rng-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-Ow2JxtOM++zMANLr6V1q2BY31i4Ubk8/IQGVK8g2w50=";
    }
    {
      name = "onemkl_sycl_sparse";
      url = "https://files.pythonhosted.org/packages/5d/88/4b7e2095f5d16ce1d8425efb2bf0126dd60ef659e24619bdeea85c10ef74/onemkl_sycl_sparse-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-Sife2BqzgmqDncHOZuRMu3OfD3EJodzNGqd8yaZmPQg=";
    }
    {
      name = "impi_rt";
      url = "https://files.pythonhosted.org/packages/ce/29/496c69c70a5645eaa2dfe230ee9ab7dcee041a64f7a6555c515512530495/impi_rt-2021.17.2-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-8QIGAQRW135bVb8ZBFy8LqxG0kA3NTlBak/s2KUlMMg=";
    }
    {
      name = "dpcpp_cpp_rt";
      url = "https://files.pythonhosted.org/packages/e9/d2/edcbe7995ae3fe18f709c8ae122da8978aee1311c3e8d3bdcd23fc684e69/dpcpp_cpp_rt-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-cITPpd1Dxgk4m2dhWyEsX19jO0S3naEoGJPb40Wfji0=";
    }
    {
      name = "tcmlib";
      url = "https://files.pythonhosted.org/packages/a1/a4/38e8b5a27b66ab286168ba6c449771ed71d71ec76524e7f12401474a5151/tcmlib-1.4.1-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-DVvZjbSNMb7H/tulwjWZv5rkPHAW1MOUbSUkLTIM7ok=";
    }
    {
      name = "tbb";
      url = "https://files.pythonhosted.org/packages/08/59/8d381a2cfe8d36c4f4ff9f94769ff2809bfc16014d888360b0e24c7e5c6b/tbb-2022.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-U0wInsprzxSECGhMFWIp1fCcAdMjtoWxVzn0GYW1Kdc=";
    }
    {
      name = "umf";
      url = "https://files.pythonhosted.org/packages/4c/b9/fe8c54eab5a3fdfdff1839d9299d90bfce5c467186b5c9ff9fd95d55ad64/umf-1.0.3-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-yJwJdNrtMKHKx3+3zl/5FA0Xjihrst2Rt1FIbV0KZbA=";
    }

    # ---- triton-xpu ----
    {
      name = "triton_xpu";
      url = "https://download-r2.pytorch.org/whl/triton_xpu-3.7.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
      hash = "sha256-T9rtG6/FHTooNGVqNCCmaGp06iJlCHZaSb8V1Y/zqTA=";
    }

    # ---- torch + torchvision (cp313, +xpu) ----
    {
      name = "torch";
      url = "https://download-r2.pytorch.org/whl/xpu/torch-2.12.0%2Bxpu-cp313-cp313-linux_x86_64.whl";
      hash = "sha256-VvdOfGwJbhp6whXree5ZC3ZL4/u6j0/rwUW8pHGUoIM=";
    }
    {
      name = "torchvision";
      url = "https://download-r2.pytorch.org/whl/xpu/torchvision-0.27.0%2Bxpu-cp313-cp313-manylinux_2_28_x86_64.whl";
      hash = "sha256-i8fTdRXOoYr0w4nV/eWLGp12sBXy2H5KfcYq1QscwgA=";
    }
  ];

  wheelSrcs = map (w: fetchurl { inherit (w) url hash; }) wheels;

  combined = stdenv.mkDerivation {
    pname = "intel-xpu-runtime";
    version = "2025.3.2";

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
