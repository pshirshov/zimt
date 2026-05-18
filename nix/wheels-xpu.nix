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
  wheel = { pname, version, url, hash ? fakeHash, deps ? [] }:
    buildPythonPackage {
      inherit pname version;
      format = "wheel";
      src = fetchurl { inherit url hash; };
      propagatedBuildInputs = deps;
      doCheck = false;
      pythonImportsCheck = [];
    };
in
[
  # ---- Intel oneAPI / SYCL runtime libraries ----
  (wheel {
    pname = "intel-cmplr-lib-rt"; version = "2025.3.2";
    url = "https://files.pythonhosted.org/packages/source/intel_cmplr_lib_rt/intel_cmplr_lib_rt-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-xBPvwbLd2Kh56VGLy8N1VBuLYzlRsoJWhFcdgzrqJq0=";
  })
  (wheel {
    pname = "intel-cmplr-lib-ur"; version = "2025.3.2";
    url = "https://files.pythonhosted.org/packages/source/intel_cmplr_lib_ur/intel_cmplr_lib_ur-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-F1zbD7ylw+8NsXbxy9i48sNtM+t3TPHQz6BAT3SN9jQ=";
  })
  (wheel {
    pname = "intel-cmplr-lic-rt"; version = "2025.3.2";
    url = "https://files.pythonhosted.org/packages/source/intel_cmplr_lic_rt/intel_cmplr_lic_rt-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-hzD7HchHQKSVYDQDUxuSRaxH0tNWOoU6exoN/jqBsrs=";
  })
  (wheel {
    pname = "intel-opencl-rt"; version = "2025.3.2";
    url = "https://files.pythonhosted.org/packages/source/intel_opencl_rt/intel_opencl_rt-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-KtxqZFDKkbCgANZEPpH1ZeKTI6DxX0Yte+VB8HaoxTY=";
  })
  (wheel {
    pname = "intel-openmp"; version = "2025.3.2";
    url = "https://files.pythonhosted.org/packages/source/intel_openmp/intel_openmp-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-LeXxL7KC5+J1cjU5y6M1zhgx5TP5SxxIQGJWnhvFc5k=";
  })
  (wheel {
    pname = "intel-pti"; version = "0.16.0";
    url = "https://files.pythonhosted.org/packages/source/intel_pti/intel_pti-0.16.0-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-RS5ER5pXILP+GfAPoBveopWGoaJaVXJ4hXqwo9EceFI=";
  })
  (wheel {
    pname = "intel-sycl-rt"; version = "2025.3.2";
    url = "https://files.pythonhosted.org/packages/source/intel_sycl_rt/intel_sycl_rt-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-OxnZ9ktWgEp7xYqR/pXarPELyQa5TOPFtkFhUzZ9+XI=";
  })
  (wheel {
    pname = "mkl"; version = "2025.3.1";
    url = "https://files.pythonhosted.org/packages/source/mkl/mkl-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-2zHln6No3U+kW0lDUfS34OYgSwjX2yeDYRjE4TcOsBE=";
  })
  (wheel {
    pname = "oneccl"; version = "2021.17.2";
    url = "https://files.pythonhosted.org/packages/source/oneccl/oneccl-2021.17.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-p3wsVu66HHg3mlMKK6K70zQH9kQLREdvgyktFpQQA3M=";
  })
  (wheel {
    pname = "oneccl-devel"; version = "2021.17.2";
    url = "https://files.pythonhosted.org/packages/source/oneccl_devel/oneccl_devel-2021.17.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-bNqvEvBPqOt5NPOyPCAkEP7OeuBd/TuMxCGXAXJvGZk=";
  })
  (wheel {
    pname = "onemkl-license"; version = "2025.3.1";
    url = "https://files.pythonhosted.org/packages/source/onemkl_license/onemkl_license-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-tiY6Ab8pwE6mg+qxcS62t0wri3InIvRv2QeE7Yd4wJM=";
  })
  (wheel {
    pname = "onemkl-sycl-blas"; version = "2025.3.1";
    url = "https://files.pythonhosted.org/packages/source/onemkl_sycl_blas/onemkl_sycl_blas-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-uVbBTbRJkFec83brcrpgELDIOuJ/wTL2Z1KxszfZ5yg=";
  })
  (wheel {
    pname = "onemkl-sycl-dft"; version = "2025.3.1";
    url = "https://files.pythonhosted.org/packages/source/onemkl_sycl_dft/onemkl_sycl_dft-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-OfQ6M00Fq1mJsizrRiQ+rM5FcdAWyfZ5Ixef9OQnuIo=";
  })
  (wheel {
    pname = "onemkl-sycl-lapack"; version = "2025.3.1";
    url = "https://files.pythonhosted.org/packages/source/onemkl_sycl_lapack/onemkl_sycl_lapack-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-Ng/GNrwIMtWi7i08T0QBRexD2UwXCjUdWIqa96hISc4=";
  })
  (wheel {
    pname = "onemkl-sycl-rng"; version = "2025.3.1";
    url = "https://files.pythonhosted.org/packages/source/onemkl_sycl_rng/onemkl_sycl_rng-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-Ow2JxtOM++zMANLr6V1q2BY31i4Ubk8/IQGVK8g2w50=";
  })
  (wheel {
    pname = "onemkl-sycl-sparse"; version = "2025.3.1";
    url = "https://files.pythonhosted.org/packages/source/onemkl_sycl_sparse/onemkl_sycl_sparse-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-Sife2BqzgmqDncHOZuRMu3OfD3EJodzNGqd8yaZmPQg=";
  })
  (wheel {
    pname = "impi-rt"; version = "2021.17.2";
    url = "https://files.pythonhosted.org/packages/source/impi_rt/impi_rt-2021.17.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-8QIGAQRW135bVb8ZBFy8LqxG0kA3NTlBak/s2KUlMMg=";
  })
  (wheel {
    pname = "dpcpp-cpp-rt"; version = "2025.3.2";
    url = "https://files.pythonhosted.org/packages/source/dpcpp_cpp_rt/dpcpp_cpp_rt-2025.3.2-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-cITPpd1Dxgk4m2dhWyEsX19jO0S3naEoGJPb40Wfji0=";
  })
  (wheel {
    pname = "tcmlib"; version = "1.4.1";
    url = "https://files.pythonhosted.org/packages/source/tcmlib/tcmlib-1.4.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-DVvZjbSNMb7H/tulwjWZv5rkPHAW1MOUbSUkLTIM7ok=";
  })
  (wheel {
    pname = "tbb"; version = "2022.3.1";
    url = "https://files.pythonhosted.org/packages/source/tbb/tbb-2022.3.1-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-U0wInsprzxSECGhMFWIp1fCcAdMjtoWxVzn0GYW1Kdc=";
  })
  (wheel {
    pname = "umf"; version = "1.0.3";
    url = "https://files.pythonhosted.org/packages/source/umf/umf-1.0.3-py2.py3-none-manylinux_2_28_x86_64.whl";
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
