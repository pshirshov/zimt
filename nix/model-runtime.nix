# Runtime dependencies required by the Qwen-Image 2.1 Diffusers revision.
# Override the package set so transitive users share the same versions.
{ lib, fetchPypi, stdenv, autoPatchelfHook, blackForChecks }:
pySelf: _pySuper:
let
  wheel = { pname, version, source, dependencies, imports }:
    pySelf.buildPythonPackage {
      inherit pname version dependencies;
      format = "wheel";
      src = fetchPypi ({ inherit pname version; format = "wheel"; dist = source.python; } // source);
      doCheck = false;
      pythonImportsCheck = imports;
    };
  safetensorsSources = {
    x86_64-linux = {
      platform = "manylinux_2_17_x86_64.manylinux2014_x86_64";
      sha256 = "fd6f3f93c9a0a7cc2788ee63fb763353d4bd2e89b0751bc78fcf7dda00bea774";
    };
    aarch64-linux = {
      platform = "manylinux_2_17_aarch64.manylinux2014_aarch64";
      sha256 = "7a46e5ff292c356d6991e60942ba7f79817682d3a2cef0702136448cb9c4d235";
    };
    x86_64-darwin = {
      platform = "macosx_10_12_x86_64";
      sha256 = "c554f85858e05226d3c2828e32395e677434685d6d94594a41643361c5e837f0";
    };
    aarch64-darwin = {
      platform = "macosx_11_0_arm64";
      sha256 = "c80201d22cbf405b80647a60ada77bba06c8fba2da2743ba1e89cdcc39a81f25";
    };
  };
in {
  # Black is a transitive check tool, not a Zimt runtime dependency. Keep
  # its original Click closure; its test doubles predate Click's abstract API.
  black = blackForChecks;
  click = wheel {
    pname = "click";
    # httpx2's checks still use APIs deprecated by Click 8.5.
    version = "8.4.2";
    source = {
      python = "py3";
      sha256 = "e6f9f66136c816745b9d65817da91d61d957fb16e02e4dcd0552553c5a197b76";
    };
    dependencies = [];
    imports = [ "click" ];
  };
  typer = wheel {
    pname = "typer";
    version = "0.27.2";
    source = {
      python = "py3";
      sha256 = "b3a5fc4342d5fc8fda8fc3010b1cf117e9249aab7fae800c2eff62fd3842d97d";
    };
    dependencies = with pySelf; [ shellingham rich annotated-doc ];
    imports = [ "typer" ];
  };
  huggingface-hub = wheel {
    pname = "huggingface_hub";
    version = "1.32.0";
    source = {
      python = "py3";
      sha256 = "b0c7c80561969d9cdacdd55fce67ba9584cca0b9d4ea80957a3a5c1445fac5c8";
    };
    dependencies = with pySelf; [
      click filelock fsspec hf-xet httpx packaging pyyaml tqdm typing-extensions
    ];
    imports = [ "huggingface_hub" ];
  };
  accelerate = wheel {
    pname = "accelerate";
    version = "1.15.0";
    source = {
      python = "py3";
      sha256 = "97eacca0b73e45cb867dbf8c5d5d4dc32219544300e0c8992c7334dc2ef33cec";
    };
    dependencies = with pySelf; [ numpy packaging psutil pyyaml torch huggingface-hub safetensors ];
    imports = [ "accelerate" ];
  };
  peft = wheel {
    pname = "peft";
    version = "0.21.0";
    source = {
      python = "py3";
      sha256 = "b64eb75fd9dece7401c70e675b8d9de024993b70483691b41c62876f0c7809b7";
    };
    dependencies = with pySelf; [
      numpy packaging psutil pyyaml torch transformers tqdm accelerate safetensors huggingface-hub
    ];
    imports = [ "peft" ];
  };
  pyelftools = wheel {
    pname = "pyelftools";
    version = "0.33";
    source = {
      python = "py3";
      sha256 = "f215ad5f47d3f1373a21496a6c9e0707c622840d0622f23ff7ce08678b020036";
    };
    dependencies = [];
    imports = [ "elftools" ];
  };
  # Use the same release wheel exercised by the XPU checks, without rebuilding
  # nixpkgs's CPU-only torch just to run safetensors's upstream check suite.
  safetensors = pySelf.buildPythonPackage rec {
    pname = "safetensors";
    version = "0.8.0";
    format = "wheel";
    src = fetchPypi ({
      inherit pname version;
      format = "wheel";
      dist = "cp310";
      python = "cp310";
      abi = "abi3";
    } // safetensorsSources.${stdenv.hostPlatform.system});
    nativeBuildInputs = lib.optional stdenv.hostPlatform.isLinux autoPatchelfHook;
    buildInputs = lib.optional stdenv.hostPlatform.isLinux stdenv.cc.cc.lib;
    doCheck = false;
    pythonImportsCheck = [ "safetensors" ];
  };
}
