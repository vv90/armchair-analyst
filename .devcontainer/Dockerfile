FROM mcr.microsoft.com/devcontainers/base:debian

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    xz-utils \
    direnv \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -m 0755 /nix && chown vscode:vscode /nix

USER vscode

RUN curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install -o /tmp/install-nix.sh \
    && sh /tmp/install-nix.sh --no-daemon \
    && rm /tmp/install-nix.sh

RUN mkdir -p /home/vscode/.config/nix \
    && printf 'experimental-features = nix-command flakes\n' > /home/vscode/.config/nix/nix.conf

ENV PATH="/home/vscode/.nix-profile/bin:/home/vscode/.local/state/nix/profile/bin:/nix/var/nix/profiles/default/bin:${PATH}"

RUN nix profile install nixpkgs#nix-direnv \
    && mkdir -p /home/vscode/.config/direnv \
    && printf 'source $HOME/.nix-profile/share/nix-direnv/direnvrc\n' > /home/vscode/.config/direnv/direnvrc \
    && printf 'eval "$(direnv hook bash)"\n' >> /home/vscode/.bashrc

USER root

RUN chown -R vscode:vscode /nix /home/vscode

USER vscode

WORKDIR /workspace
