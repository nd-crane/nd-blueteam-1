{python3}: let
  inherit
    (python3.pkgs)
    buildPythonApplication
    pythonOlder
    opencv4
    pytorch-bin
    torchvision-bin
    ;
in
  buildPythonApplication rec {
    pname = "blue-team-nd-1";
    version = "0.0.1";
    format = "pyproject";
    disabled = pythonOlder "3.8";

    HOME = ".";

    src = ../.;

    propagatedBuildInputs = [
      opencv4
      pytorch-bin
      torchvision-bin
    ];
  }
