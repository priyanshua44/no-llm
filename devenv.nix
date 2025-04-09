{ pkgs, lib, config, inputs, ... }:

{
  packages = [ 
    pkgs.git 
  ];
  languages.python = {
    enable = true;
    uv.enable = true;
  };

  pre-commit.hooks = {
    shellcheck.enable = true;
    convco.enable = true;
  };
  difftastic.enable = true;
  cachix.enable = true;
}

