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
{ python
, fetchurl
, buildPythonPackage
}:

let
  lib = python.pkgs.lib or (import <nixpkgs> {}).lib;
  fakeHash = lib.fakeHash;

  # Helper: a single Python wheel package. Pip-style names; no deps listed
  # because the Python env in package.nix layers nixpkgs deps separately.
  # The runtime-deps check reads wheel metadata before the full env exists,
  # so it must stay disabled for these standalone wheel wrappers.
  wheel = { pname, version, url, hash ? fakeHash, deps ? [] }:
    buildPythonPackage {
      inherit pname version;
      format = "wheel";
      src = fetchurl { inherit url hash; };
      propagatedBuildInputs = deps;
      dontCheckRuntimeDeps = true;
      doCheck = false;
      pythonImportsCheck = [];
    };
in
[
  # ---- Intel oneAPI / SYCL runtime libraries ----
  (wheel {
    pname = "intel-cmplr-lib-rt"; version = "2025.3.2";
    url = "https://files.pythonhosted.org/packages/1b/d7/ffb7e58ac260737b5076e2738ce198468a4efb3ba1885bced038801c387a/intel_cmplr_lib_rt-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-xBPvwbLd2Kh56VGLy8N1VBuLYzlRsoJWhFcdgzrqJq0=";
  })
  (wheel {
    pname = "intel-cmplr-lib-ur"; version = "2025.3.2";
    url = "https://files.pythonhosted.org/packages/49/7e/bc668ee301350964b7be980db7d527a54a61b89a749d09bb430b4598ed3b/intel_cmplr_lib_ur-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-F1zbD7ylw+8NsXbxy9i48sNtM+t3TPHQz6BAT3SN9jQ=";
  })
  (wheel {
    pname = "intel-cmplr-lic-rt"; version = "2025.3.2";
    url = "https://files.pythonhosted.org/packages/c1/27/bbadf924bf134143895fc197701951d9d28c77bd1cfbabd5e1dfb9b90b8b/intel_cmplr_lic_rt-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-hzD7HchHQKSVYDQDUxuSRaxH0tNWOoU6exoN/jqBsrs=";
  })
  (wheel {
    pname = "intel-opencl-rt"; version = "2025.3.2";
    url = "https://files.pythonhosted.org/packages/40/6b/51459a9a6ac585ab8ca1accc58a580a23b91c5272a63ccf419c8f9d52f37/intel_opencl_rt-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-KtxqZFDKkbCgANZEPpH1ZeKTI6DxX0Yte+VB8HaoxTY=";
  })
  (wheel {
    pname = "intel-openmp"; version = "2025.3.2";
    url = "https://files.pythonhosted.org/packages/91/99/5ee1e9ae85ed3d10517d17d5a3a924fecc15b116385045a0a89eb2d5bc82/intel_openmp-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-LeXxL7KC5+J1cjU5y6M1zhgx5TP5SxxIQGJWnhvFc5k=";
  })
  (wheel {
    pname = "intel-pti"; version = "0.16.0";
    url = "https://files.pythonhosted.org/packages/af/85/dee48118c530d9574f683f8cf3a7ad576a23f060a520335c9284ff6ba65b/intel_pti-0.16.0-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-RS5ER5pXILP+GfAPoBveopWGoaJaVXJ4hXqwo9EceFI=";
  })
  (wheel {
    pname = "intel-sycl-rt"; version = "2025.3.2";
    url = "https://files.pythonhosted.org/packages/f7/e4/047a0b42f8240a9c70f3ac120479aa1996f99cfe9fd97afb025d6bbd89a8/intel_sycl_rt-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-OxnZ9ktWgEp7xYqR/pXarPELyQa5TOPFtkFhUzZ9+XI=";
  })
  (wheel {
    pname = "mkl"; version = "2025.3.1";
    url = "https://files.pythonhosted.org/packages/b3/ee/76755ca0ec9626835e0d024c369b968f24eadce2106a7884404720670623/mkl-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-2zHln6No3U+kW0lDUfS34OYgSwjX2yeDYRjE4TcOsBE=";
  })
  (wheel {
    pname = "oneccl"; version = "2021.17.2";
    url = "https://files.pythonhosted.org/packages/73/9b/2932b6b128924ba96712ea1c807c1618b9963518eb8ec80cf834ffc3c684/oneccl-2021.17.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-p3wsVu66HHg3mlMKK6K70zQH9kQLREdvgyktFpQQA3M=";
  })
  (wheel {
    pname = "oneccl-devel"; version = "2021.17.2";
    url = "https://files.pythonhosted.org/packages/4a/a1/4fad3108825e2d65d812ba69a9fd3664181cfe8860e49110c92431d1629f/oneccl_devel-2021.17.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-bNqvEvBPqOt5NPOyPCAkEP7OeuBd/TuMxCGXAXJvGZk=";
  })
  (wheel {
    pname = "onemkl-license"; version = "2025.3.1";
    url = "https://files.pythonhosted.org/packages/3e/1d/7acbedb07bf4c71cc499527c25a3ef60bf83ed41b8918e986ed7a4573bd4/onemkl_license-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-tiY6Ab8pwE6mg+qxcS62t0wri3InIvRv2QeE7Yd4wJM=";
  })
  (wheel {
    pname = "onemkl-sycl-blas"; version = "2025.3.1";
    url = "https://files.pythonhosted.org/packages/51/17/497d29cd13029f4835383d95e644d1602dafa9b7887d298ba9ea77734dce/onemkl_sycl_blas-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-uVbBTbRJkFec83brcrpgELDIOuJ/wTL2Z1KxszfZ5yg=";
  })
  (wheel {
    pname = "onemkl-sycl-dft"; version = "2025.3.1";
    url = "https://files.pythonhosted.org/packages/41/ae/46fe3ca4fcf715cfef35b239abe705d32f355c3be4b9e94aca782a4720ae/onemkl_sycl_dft-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-OfQ6M00Fq1mJsizrRiQ+rM5FcdAWyfZ5Ixef9OQnuIo=";
  })
  (wheel {
    pname = "onemkl-sycl-lapack"; version = "2025.3.1";
    url = "https://files.pythonhosted.org/packages/24/f6/f4e38bb1a81fbda3afa2d79aa95b5a5d73c838977c1f86fbf73e7dafe676/onemkl_sycl_lapack-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-Ng/GNrwIMtWi7i08T0QBRexD2UwXCjUdWIqa96hISc4=";
  })
  (wheel {
    pname = "onemkl-sycl-rng"; version = "2025.3.1";
    url = "https://files.pythonhosted.org/packages/60/f9/1f3b6ce37848c721462b4f30de08482dc27d6ffbcbba621b61ef53e28d3c/onemkl_sycl_rng-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-Ow2JxtOM++zMANLr6V1q2BY31i4Ubk8/IQGVK8g2w50=";
  })
  (wheel {
    pname = "onemkl-sycl-sparse"; version = "2025.3.1";
    url = "https://files.pythonhosted.org/packages/5d/88/4b7e2095f5d16ce1d8425efb2bf0126dd60ef659e24619bdeea85c10ef74/onemkl_sycl_sparse-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-Sife2BqzgmqDncHOZuRMu3OfD3EJodzNGqd8yaZmPQg=";
  })
  (wheel {
    pname = "impi-rt"; version = "2021.17.2";
    url = "https://files.pythonhosted.org/packages/ce/29/496c69c70a5645eaa2dfe230ee9ab7dcee041a64f7a6555c515512530495/impi_rt-2021.17.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-8QIGAQRW135bVb8ZBFy8LqxG0kA3NTlBak/s2KUlMMg=";
  })
  (wheel {
    pname = "dpcpp-cpp-rt"; version = "2025.3.2";
    url = "https://files.pythonhosted.org/packages/e9/d2/edcbe7995ae3fe18f709c8ae122da8978aee1311c3e8d3bdcd23fc684e69/dpcpp_cpp_rt-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-cITPpd1Dxgk4m2dhWyEsX19jO0S3naEoGJPb40Wfji0=";
  })
  (wheel {
    pname = "tcmlib"; version = "1.4.1";
    url = "https://files.pythonhosted.org/packages/a1/a4/38e8b5a27b66ab286168ba6c449771ed71d71ec76524e7f12401474a5151/tcmlib-1.4.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-DVvZjbSNMb7H/tulwjWZv5rkPHAW1MOUbSUkLTIM7ok=";
  })
  (wheel {
    pname = "tbb"; version = "2022.3.1";
    url = "https://files.pythonhosted.org/packages/08/59/8d381a2cfe8d36c4f4ff9f94769ff2809bfc16014d888360b0e24c7e5c6b/tbb-2022.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-U0wInsprzxSECGhMFWIp1fCcAdMjtoWxVzn0GYW1Kdc=";
  })
  (wheel {
    pname = "umf"; version = "1.0.3";
    url = "https://files.pythonhosted.org/packages/4c/b9/fe8c54eab5a3fdfdff1839d9299d90bfce5c467186b5c9ff9fd95d55ad64/umf-1.0.3-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-yJwJdNrtMKHKx3+3zl/5FA0Xjihrst2Rt1FIbV0KZbA=";
  })

  # ---- triton-xpu ----
  (wheel {
    pname = "triton-xpu"; version = "3.7.1";
    url = "https://download-r2.pytorch.org/whl/triton_xpu-3.7.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    hash = "sha256-T9rtG6/FHTooNGVqNCCmaGp06iJlCHZaSb8V1Y/zqTA=";
  })

  # ---- torch + torchvision (cp313, +xpu) ----
  (wheel {
    pname = "torch"; version = "2.12.0+xpu";
    url = "https://download-r2.pytorch.org/whl/xpu/torch-2.12.0%2Bxpu-cp313-cp313-linux_x86_64.whl";
    hash = "sha256-VvdOfGwJbhp6whXree5ZC3ZL4/u6j0/rwUW8pHGUoIM=";
  })
  (wheel {
    pname = "torchvision"; version = "0.27.0+xpu";
    url = "https://download-r2.pytorch.org/whl/xpu/torchvision-0.27.0%2Bxpu-cp313-cp313-manylinux_2_28_x86_64.whl";
    hash = "sha256-i8fTdRXOoYr0w4nV/eWLGp12sBXy2H5KfcYq1QscwgA=";
  })
]
