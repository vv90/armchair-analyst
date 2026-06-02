{
  description = "Rust dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { nixpkgs, rust-overlay, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system:
          f system
        );

      pkgsFor = system:
        import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };

      nativeRustTarget = system:
        {
          x86_64-linux = "x86_64-unknown-linux-gnu";
          aarch64-linux = "aarch64-unknown-linux-gnu";
          x86_64-darwin = "x86_64-apple-darwin";
          aarch64-darwin = "aarch64-apple-darwin";
        }.${system};

    in {
      devShells = forAllSystems (system:
        let
          pkgs = pkgsFor system;

          rustToolchain = pkgs.rust-bin.stable.latest.default.override {
            extensions = [
              "rust-src"
              "rustfmt"
              "clippy"
            ];

            # These are Rust compilation targets, not Nix systems.
            # The native target is usually already included, but listing it is harmless.
            targets = [
              (nativeRustTarget system)
              "wasm32-unknown-unknown"
            ];
          };
        in {
          default = pkgs.mkShell {
            packages = with pkgs; [
              rustToolchain
              rust-analyzer
              nil
              pkg-config
              openssl
              git
              codex
            ];
          };
        });
    };
}
