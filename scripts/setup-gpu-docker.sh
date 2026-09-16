#!/usr/bin/env bash
# =============================================================================
#  setup-gpu-docker.sh — provision a machine for GPU containers:
#     NVIDIA driver + Docker Engine + NVIDIA Container Toolkit.
#
#  This is everything that has to exist before the SGLang dev container can
#  start. Once this script is done (and you have rebooted / re-logged in if it
#  says so), the launch command is:
#
#     docker run -itd --name sglang_dev \
#         --gpus all --shm-size 32g --ipc=host --network=host --privileged \
#         -v <repo>:/sgl-workspace/sglang \
#         -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
#         lmsysorg/sglang:dev /bin/zsh
#
#  Reproduces (idempotently) the standard vendor instructions:
#     sudo apt-get install linux-headers-$(uname -r) nvidia-driver-<N>-server-open
#     # Docker's apt repo   -> docker-ce + cli + containerd + buildx + compose
#     # NVIDIA's apt repo   -> nvidia-container-toolkit
#     sudo nvidia-ctk runtime configure --runtime=docker
#     sudo usermod -aG docker $USER
#
#  Run it as your normal user; sudo is used only for apt-get, usermod,
#  systemctl and writes under /etc. Running the whole script under sudo also
#  works — $SUDO_USER is what gets added to the 'docker' group.
#
#  Safe to re-run: every step checks first and skips what is already in place.
#
#  Usage:
#     ./setup-gpu-docker.sh [options]
#
#  Options:
#     --driver-branch <N>  NVIDIA driver branch (default 580). CUDA 13 images
#                          such as lmsysorg/sglang:dev require >= 580.
#     --skip-driver        Don't touch the NVIDIA driver
#     --skip-docker        Don't install Docker Engine
#     --skip-toolkit       Don't install the NVIDIA Container Toolkit
#     --no-docker-group    Don't add the user to the 'docker' group
#     --no-smoke-test      Don't run `docker run --gpus all ... nvidia-smi`
#     --reboot             Reboot at the end if the new driver needs it
#     --dry-run            Print what would change; change nothing
#     -h, --help           Show this help
#
#  References:
#     https://docs.docker.com/engine/install/ubuntu/
#     https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
#     https://docs.sglang.ai/docs/developer_guide/development_guide_using_docker
# =============================================================================

# Re-exec under bash if someone runs us with `sh setup-gpu-docker.sh`.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# ----------------------------------------------------------------- config ---
DOCKER_GPG_URL="https://download.docker.com/linux/ubuntu/gpg"
DOCKER_REPO_URL="https://download.docker.com/linux/ubuntu"
DOCKER_KEYRING="/etc/apt/keyrings/docker.asc"
DOCKER_LIST="/etc/apt/sources.list.d/docker.list"
DOCKER_PACKAGES=(docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin)

NVCT_GPG_URL="https://nvidia.github.io/libnvidia-container/gpgkey"
NVCT_LIST_URL="https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list"
NVCT_KEYRING="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
NVCT_LIST="/etc/apt/sources.list.d/nvidia-container-toolkit.list"

# lmsysorg/sglang:dev is built on CUDA 13.0 (docker/Dockerfile: ARG
# CUDA_VERSION=13.0.3). The container runtime honours the image's
# NVIDIA_REQUIRE_CUDA label and refuses to start it on an older driver.
DRIVER_BRANCH=580
MIN_DRIVER_MAJOR=580

# Any image works: the container toolkit injects nvidia-smi from the host
# driver, so this needs no CUDA userspace and stays tiny.
SMOKE_IMAGE="ubuntu:24.04"

SKIP_DRIVER=0
SKIP_DOCKER=0
SKIP_TOOLKIT=0
DO_DOCKER_GROUP=1
DO_SMOKE_TEST=1
DO_REBOOT=0
DRY_RUN=0

APT_STALE=1        # 1 = `apt-get update` has not run yet in this invocation
NEED_REBOOT=0
NEED_RELOGIN=0
CHANGED=0
WORKDIR=""

# ----------------------------------------------------------------- output ---
if [ -t 1 ]; then
  C_B=$'\033[1m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_R=$'\033[31m'; C_0=$'\033[0m'
else
  C_B=''; C_G=''; C_Y=''; C_R=''; C_0=''
fi
step() { printf '\n%s==> %s%s\n' "$C_B" "$*" "$C_0"; }
info() { printf '    %s\n' "$*"; }
ok()   { printf '    %s+%s %s\n' "$C_G" "$C_0" "$*"; }
warn() { printf '    %s!%s %s\n' "$C_Y" "$C_0" "$*" >&2; }
die()  { printf '\n%sError:%s %s\n' "$C_R" "$C_0" "$*" >&2; exit 1; }

# One private scratch dir for the whole run, removed on exit. mktempfile() is
# always called from a command substitution, so it cannot register anything in a
# parent-shell array - the directory is what makes cleanup reliable.
cleanup() {
  [ -n "$WORKDIR" ] && [ -d "$WORKDIR" ] && rm -rf "$WORKDIR"
  return 0
}
trap cleanup EXIT
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/setup-gpu-docker.XXXXXX")

_tmpseq=0
mktempfile() {
  _tmpseq=$((_tmpseq + 1))
  local f="$WORKDIR/f$$.$_tmpseq"
  : > "$f"
  printf '%s\n' "$f"
}

usage() { sed -n '2,/^# ===\+$/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; }

# Every mutating command goes through run(), which is what makes --dry-run
# honest. Pipelines are therefore split into a harmless fetch (curl/sed into
# $WORKDIR) plus a simple `install`/`apt-get` that run() can intercept.
run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '    %s[dry-run]%s %s\n' "$C_Y" "$C_0" "$*"
    return 0
  fi
  "$@"
}

# ------------------------------------------------------------------- args ---
while [ $# -gt 0 ]; do
  case "$1" in
    --driver-branch)
      [ $# -ge 2 ] || die "--driver-branch needs a value (e.g. 580)"
      DRIVER_BRANCH="$2"; shift 2 ;;
    --driver-branch=*) DRIVER_BRANCH="${1#*=}"; shift ;;
    --skip-driver)     SKIP_DRIVER=1; shift ;;
    --skip-docker)     SKIP_DOCKER=1; shift ;;
    --skip-toolkit)    SKIP_TOOLKIT=1; shift ;;
    --no-docker-group) DO_DOCKER_GROUP=0; shift ;;
    --no-smoke-test)   DO_SMOKE_TEST=0; shift ;;
    --reboot)          DO_REBOOT=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    *)                 die "Unknown option: $1  (try --help)" ;;
  esac
done

[[ "$DRIVER_BRANCH" =~ ^[0-9]+$ ]] || die "--driver-branch must be a number like 580"

# -------------------------------------------------------------- preflight ---
# Unlike setup-shell.sh nothing here is installed into $HOME, so running the
# whole script under sudo is fine - we just have to add the *real* user to the
# 'docker' group rather than root.
if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
  CURRENT_USER="${SUDO_USER:-root}"
elif command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
  CURRENT_USER="${USER:-$(id -un)}"
else
  SUDO=""
  CURRENT_USER="${USER:-$(id -un)}"
fi

have_root() { [ "$(id -u)" -eq 0 ] || [ -n "$SUDO" ]; }

# The sglang checkout that lives next to this script. pwd -P resolves the
# symlink, so ~/MLSys-Learn/... and /lambda/nfs/MLSys-Learn/... agree.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SGLANG_REPO="$SCRIPT_DIR/sglang"

# ------------------------------------------------------------ apt helpers ---
# Is $1 installable from the configured sources?
apt_has() {
  local cand
  cand=$(apt-cache policy "$1" 2>/dev/null | awk '/Candidate:/ {print $2; exit}')
  [ -n "$cand" ] && [ "$cand" != "(none)" ]
}

apt_update_once() {
  [ "$APT_STALE" -eq 1 ] || return 0
  run $SUDO env DEBIAN_FRONTEND=noninteractive apt-get update -qq
  APT_STALE=0
}

# Install only the packages that are actually missing, so a re-run is quiet.
apt_install() {
  local p want=()
  for p in "$@"; do
    dpkg -s "$p" >/dev/null 2>&1 || want+=("$p")
  done
  [ "${#want[@]}" -eq 0 ] && return 0
  apt_update_once
  info "apt-get install ${want[*]}"
  run $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y "${want[@]}"
  CHANGED=1
}

# curl + gpg are needed to add either vendor repo.
ensure_repo_prereqs() {
  apt_install ca-certificates curl gnupg
}

# ------------------------------------------------------- 1. NVIDIA driver ---
gpu_present() {
  command -v lspci >/dev/null 2>&1 || return 0   # can't tell; assume yes
  lspci 2>/dev/null | grep -qiE '(vga|3d|display).*nvidia'
}

driver_ok() { nvidia-smi -L >/dev/null 2>&1; }

driver_version() {
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1
}

# 0 = Secure Boot on, 1 = off, 2 = unknown. DKMS modules are unsigned, so a
# machine with Secure Boot on needs MOK enrollment before they will load.
secure_boot_state() {
  [ -d /sys/firmware/efi ] || return 1
  if command -v mokutil >/dev/null 2>&1; then
    mokutil --sb-state 2>/dev/null | grep -qi 'enabled' && return 0
    return 1
  fi
  return 2
}

# Datacenter parts (H100 & friends) want the -server-open flavour; fall back
# through the other flavours so this also works on a workstation GPU.
pick_driver_package() {
  local branch="$1" p
  for p in "nvidia-driver-${branch}-server-open" \
           "nvidia-driver-${branch}-open" \
           "nvidia-driver-${branch}-server" \
           "nvidia-driver-${branch}"; do
    if apt_has "$p"; then printf '%s\n' "$p"; return 0; fi
  done
  return 1
}

list_driver_branches() {
  apt-cache search --names-only '^nvidia-driver-[0-9]+' 2>/dev/null \
    | sed -E 's/^nvidia-driver-([0-9]+).*/\1/' | sort -un | tr '\n' ' '
}

install_driver() {
  step "NVIDIA driver"
  if [ "$SKIP_DRIVER" -eq 1 ]; then
    info "Skipped (--skip-driver)"
    return 0
  fi

  if ! gpu_present; then
    warn "No NVIDIA VGA/3D/Display controller on the PCI bus; skipping the driver."
    return 0
  fi

  if driver_ok; then
    local cur major
    cur=$(driver_version)
    major="${cur%%.*}"
    if [ -n "$major" ] && [ "$major" -ge "$MIN_DRIVER_MAJOR" ] 2>/dev/null; then
      ok "Driver $cur is already loaded"
      return 0
    fi
    warn "Driver $cur is older than $MIN_DRIVER_MAJOR; CUDA 13 images will refuse to start."
    info "Installing the $DRIVER_BRANCH branch over it."
  fi

  if [ "$DRIVER_BRANCH" -lt "$MIN_DRIVER_MAJOR" ]; then
    warn "Branch $DRIVER_BRANCH is below $MIN_DRIVER_MAJOR; lmsysorg/sglang:dev (CUDA 13) will not run on it."
  fi

  have_root || die "Need root (sudo) to install the NVIDIA driver."
  command -v apt-get >/dev/null 2>&1 || die "apt-get not found. Install the driver manually and re-run with --skip-driver."

  local sb=0
  secure_boot_state || sb=$?
  case "$sb" in
    0) warn "Secure Boot is ENABLED. The DKMS modules are unsigned and will not load"
       warn "until you enroll a MOK key (apt will prompt for one during install)." ;;
    2) warn "Could not determine the Secure Boot state (mokutil not installed)." ;;
  esac

  apt_update_once
  local pkg
  pkg=$(pick_driver_package "$DRIVER_BRANCH") || die \
    "No nvidia-driver-$DRIVER_BRANCH-* package in your apt sources. Available branches: $(list_driver_branches)"
  info "Package: $pkg"

  # DKMS builds against the running kernel's headers.
  apt_install "linux-headers-$(uname -r)"
  apt_install "$pkg"

  if [ "$DRY_RUN" -eq 1 ]; then
    NEED_REBOOT=1
    return 0
  fi

  # A fresh install can usually be brought up without rebooting: modprobe loads
  # the module, and nvidia-smi as root creates the /dev/nvidia* nodes.
  $SUDO modprobe nvidia >/dev/null 2>&1 || true
  $SUDO nvidia-smi -L >/dev/null 2>&1 || true
  if driver_ok; then
    ok "Driver $(driver_version) loaded without a reboot"
  else
    NEED_REBOOT=1
    warn "Driver installed but not loaded yet - a reboot is required."
  fi
}

# ------------------------------------------------------- 2. Docker Engine ---
add_docker_repo() {
  if [ -f "$DOCKER_KEYRING" ] && [ -f "$DOCKER_LIST" ]; then
    ok "Docker apt repo already configured"
    return 0
  fi

  local codename arch key list
  # shellcheck disable=SC1091  # /etc/os-release is generated at boot
  codename=$(. /etc/os-release && printf '%s\n' "${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}")
  [ -n "$codename" ] || die "Could not determine the Ubuntu codename from /etc/os-release."
  arch=$(dpkg --print-architecture)

  # apt reads armored keys directly as long as the file is named *.asc.
  key=$(mktempfile)
  info "Fetching $DOCKER_GPG_URL"
  curl -fsSL "$DOCKER_GPG_URL" -o "$key" || die "Failed to download Docker's GPG key."
  run $SUDO install -m 0755 -d /etc/apt/keyrings
  run $SUDO install -m 0644 "$key" "$DOCKER_KEYRING"

  list=$(mktempfile)
  printf 'deb [arch=%s signed-by=%s] %s %s stable\n' \
    "$arch" "$DOCKER_KEYRING" "$DOCKER_REPO_URL" "$codename" > "$list"
  run $SUDO install -m 0644 "$list" "$DOCKER_LIST"

  APT_STALE=1
  CHANGED=1
  ok "Added Docker's apt repo ($codename/$arch)"
}

ensure_docker_running() {
  if ! command -v systemctl >/dev/null 2>&1; then
    warn "systemctl not found; start the docker daemon yourself."
    return 0
  fi
  if systemctl is-active --quiet docker 2>/dev/null; then
    ok "dockerd is running"
    return 0
  fi
  info "Starting and enabling dockerd"
  run $SUDO systemctl enable --now docker \
    || warn "Could not start dockerd. Check: systemctl status docker"
}

add_to_docker_group() {
  if [ "$DO_DOCKER_GROUP" -eq 0 ]; then
    info "Leaving the 'docker' group alone (--no-docker-group)"
    return 0
  fi
  if [ "$CURRENT_USER" = "root" ]; then
    info "Running as root with no SUDO_USER; nobody to add to the 'docker' group"
    return 0
  fi
  if id -nG "$CURRENT_USER" 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
    ok "$CURRENT_USER is already in the 'docker' group"
    return 0
  fi
  if ! have_root; then
    warn "Need root to add $CURRENT_USER to the 'docker' group; skipping."
    return 0
  fi
  run $SUDO usermod -aG docker "$CURRENT_USER"
  NEED_RELOGIN=1
  CHANGED=1
  ok "Added $CURRENT_USER to the 'docker' group"
}

install_docker() {
  step "Docker Engine"
  if [ "$SKIP_DOCKER" -eq 1 ]; then
    info "Skipped (--skip-docker)"
    return 0
  fi

  if command -v docker >/dev/null 2>&1; then
    ok "Already installed: $(docker --version 2>/dev/null || echo 'unknown version')"
  else
    have_root || die "Need root (sudo) to install Docker."
    command -v apt-get >/dev/null 2>&1 || die "apt-get not found. Install Docker manually and re-run with --skip-docker."

    # Ubuntu's docker.io and Docker's docker-ce cannot coexist.
    if dpkg -s docker.io >/dev/null 2>&1; then
      die "Ubuntu's 'docker.io' package is installed and conflicts with docker-ce.
       Remove it (sudo apt-get remove docker.io) or re-run with --skip-docker."
    fi

    ensure_repo_prereqs
    add_docker_repo
    apt_install "${DOCKER_PACKAGES[@]}"
    ok "Installed: $(docker --version 2>/dev/null || echo docker-ce)"
  fi

  ensure_docker_running
  add_to_docker_group
}

# -------------------------------------------- 3. NVIDIA Container Toolkit ---
add_toolkit_repo() {
  if [ -f "$NVCT_KEYRING" ] && [ -f "$NVCT_LIST" ]; then
    ok "NVIDIA Container Toolkit apt repo already configured"
    return 0
  fi

  local armored key list signed
  armored=$(mktempfile)
  key=$(mktempfile)
  info "Fetching $NVCT_GPG_URL"
  curl -fsSL "$NVCT_GPG_URL" -o "$armored" || die "Failed to download NVIDIA's GPG key."
  gpg --dearmor < "$armored" > "$key" || die "Failed to dearmor NVIDIA's GPG key."
  run $SUDO install -m 0644 "$key" "$NVCT_KEYRING"

  list=$(mktempfile)
  signed=$(mktempfile)
  info "Fetching $NVCT_LIST_URL"
  curl -fsSL "$NVCT_LIST_URL" -o "$list" || die "Failed to download the toolkit sources list."
  # Upstream ships the list unsigned; point each entry at the keyring above.
  sed "s#deb https://#deb [signed-by=$NVCT_KEYRING] https://#g" "$list" > "$signed"
  run $SUDO install -m 0644 "$signed" "$NVCT_LIST"

  APT_STALE=1
  CHANGED=1
  ok "Added NVIDIA's apt repo"
}

configure_docker_runtime() {
  if ! command -v docker >/dev/null 2>&1; then
    warn "Docker is not installed; skipping runtime configuration."
    return 0
  fi
  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    [ "$DRY_RUN" -eq 1 ] && { info "[dry-run] would run: sudo nvidia-ctk runtime configure --runtime=docker"; return 0; }
    warn "nvidia-ctk not found; skipping runtime configuration."
    return 0
  fi

  if grep -q '"nvidia"' /etc/docker/daemon.json 2>/dev/null; then
    ok "Docker already has the 'nvidia' runtime registered"
    return 0
  fi

  info "Registering the 'nvidia' runtime in /etc/docker/daemon.json"
  run $SUDO nvidia-ctk runtime configure --runtime=docker \
    || die "nvidia-ctk runtime configure failed."
  if command -v systemctl >/dev/null 2>&1; then
    run $SUDO systemctl restart docker \
      || warn "Could not restart dockerd. Run: sudo systemctl restart docker"
  fi
  CHANGED=1
  ok "Registered the 'nvidia' runtime and restarted dockerd"
}

install_toolkit() {
  step "NVIDIA Container Toolkit"
  if [ "$SKIP_TOOLKIT" -eq 1 ]; then
    info "Skipped (--skip-toolkit)"
    return 0
  fi

  if command -v nvidia-ctk >/dev/null 2>&1; then
    ok "Already installed: $(nvidia-ctk --version 2>/dev/null | head -1 || echo 'unknown version')"
  else
    have_root || die "Need root (sudo) to install the NVIDIA Container Toolkit."
    command -v apt-get >/dev/null 2>&1 || die "apt-get not found. Install the toolkit manually and re-run with --skip-toolkit."
    ensure_repo_prereqs
    add_toolkit_repo
    apt_install nvidia-container-toolkit
    ok "Installed: $(nvidia-ctk --version 2>/dev/null | head -1 || echo nvidia-container-toolkit)"
  fi

  configure_docker_runtime
}

# -------------------------------------------------------------- 4. verify ---
# In the group database but not in this shell's process credentials: the
# membership is real, the shell just predates it. That is a re-login, not a
# broken daemon, and it is worth telling those two apart.
docker_group_stale() {
  id -nG "$CURRENT_USER" 2>/dev/null | tr ' ' '\n' | grep -qx docker || return 1
  if id -nG 2>/dev/null | tr ' ' '\n' | grep -qx docker; then return 1; fi
  return 0
}

docker_cmd() {
  if docker info >/dev/null 2>&1; then
    printf 'docker\n'
  elif [ -n "$SUDO" ] && $SUDO -n docker info >/dev/null 2>&1; then
    printf '%s docker\n' "$SUDO"
  else
    return 1
  fi
}

smoke_test() {
  if [ "$DO_SMOKE_TEST" -eq 0 ]; then
    info "GPU-in-container test skipped (--no-smoke-test)"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    info "GPU-in-container test skipped (--dry-run)"
    return 0
  fi
  if ! driver_ok; then
    info "GPU-in-container test skipped (the driver is not live yet)"
    return 0
  fi

  local dk
  dk=$(docker_cmd) || { warn "Cannot talk to dockerd; skipping the GPU-in-container test."; return 0; }

  info "Running: $dk run --rm --gpus all $SMOKE_IMAGE nvidia-smi"
  if $dk run --rm --gpus all "$SMOKE_IMAGE" nvidia-smi >/dev/null 2>&1; then
    ok "GPUs are visible inside containers"
  else
    warn "'--gpus all' could not see the GPUs. Re-run by hand to see the error:"
    warn "  $dk run --rm --gpus all $SMOKE_IMAGE nvidia-smi"
  fi
}

verify() {
  step "Verifying"

  if driver_ok; then
    ok "nvidia-smi: driver $(driver_version)"
    nvidia-smi -L 2>/dev/null | sed 's/^/      /'
  elif [ "$SKIP_DRIVER" -eq 1 ]; then
    info "Driver step skipped; not checking nvidia-smi"
  elif [ "$NEED_REBOOT" -eq 1 ]; then
    warn "nvidia-smi does not work yet - reboot to load the driver."
  else
    warn "nvidia-smi does not work."
  fi

  if command -v docker >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then
      ok "docker: server $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo 'running')"
    elif [ "$NEED_RELOGIN" -eq 1 ] || docker_group_stale; then
      NEED_RELOGIN=1
      info "'docker info' needs the 'docker' group to be live in this shell (see below)"
    else
      warn "'docker info' failed. Check: systemctl status docker"
    fi
  elif [ "$SKIP_DOCKER" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    warn "docker is not on PATH."
  fi

  smoke_test
}

# ---------------------------------------------------------------- summary ---
summary() {
  step "Done"
  local drv
  drv=$(driver_version || true)
  info "driver:            ${drv:-not loaded}"
  info "docker:            $(docker --version 2>/dev/null || echo 'not installed')"
  info "container toolkit: $(nvidia-ctk --version 2>/dev/null | head -1 || echo 'not installed')"

  if [ "$NEED_REBOOT" -eq 1 ]; then
    printf '\n    %sReboot to load the new driver:  sudo reboot%s\n' "$C_B" "$C_0"
  fi
  if [ "$NEED_RELOGIN" -eq 1 ]; then
    printf '    %sLog out and back in (or: newgrp docker) to use docker without sudo.%s\n' "$C_B" "$C_0"
  fi
  if [ "$CHANGED" -eq 0 ]; then
    printf '\n    %sNothing to change - everything was already in place.%s\n' "$C_B" "$C_0"
  fi

  cat <<NEXT

    Next - start the SGLang dev container. The repo is bind-mounted over the
    image's own editable install, so your local edits are live with no reinstall:

      mkdir -p "\$HOME/.cache/huggingface"
      docker run -itd --name sglang_dev \\
        --gpus all --shm-size 32g --ipc=host --network=host --privileged \\
        -v $SGLANG_REPO:/sgl-workspace/sglang \\
        -v "\$HOME/.cache/huggingface":/root/.cache/huggingface \\
        lmsysorg/sglang:dev /bin/zsh
      docker exec -it sglang_dev /bin/zsh
NEXT
}

maybe_reboot() {
  [ "$NEED_REBOOT" -eq 1 ] || return 0
  [ "$DO_REBOOT" -eq 1 ] || return 0
  if [ "$DRY_RUN" -eq 1 ]; then
    info "[dry-run] would reboot now"
    return 0
  fi
  step "Rebooting in 5s (Ctrl-C to cancel)"
  sleep 5
  $SUDO reboot
}

# ------------------------------------------------------------------- main ---
main() {
  printf '%s%s%s\n' "$C_B" \
    "setup-gpu-docker.sh - NVIDIA driver + Docker Engine + NVIDIA Container Toolkit" "$C_0"
  if [ "$DRY_RUN" -eq 1 ]; then
    warn "Dry run: nothing will be installed or changed."
  fi
  install_driver
  install_docker
  install_toolkit
  verify
  summary
  maybe_reboot
}

main "$@"
