#!/usr/bin/env bash
# =============================================================================
#  setup-shell.sh — provision a machine with zsh + Oh My Zsh +
#  zsh-autosuggestions + the latest Claude Code CLI, plus the host-side git
#  identity and an SSH key for GitHub.
#
#  Reproduces (idempotently, and with backups) the setup of this box:
#     sudo apt install zsh
#     sh -c "$(curl -fsSL .../ohmyzsh/tools/install.sh)"
#     git clone https://github.com/zsh-users/zsh-autosuggestions \
#         ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
#     # add zsh-autosuggestions to plugins=(...) in ~/.zshrc
#     curl -fsSL https://claude.ai/install.sh | bash
#     # add ~/.local/bin to PATH in ~/.zshrc
#     git config --global user.name  "Haoran Wan"
#     git config --global user.email "haoran.w@princeton.edu"
#     ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
#     # pin github.com's published ed25519 host key in ~/.ssh/known_hosts,
#     # then paste ~/.ssh/id_ed25519.pub at https://github.com/settings/ssh/new
#
#  Run it as your normal user (NOT with sudo). sudo is used only for
#  `apt-get install` and `chsh`.
#
#  Usage:
#     ./setup-shell.sh [options]
#
#  Options:
#     --claude-channel <ch>  stable (default) | latest | X.Y.Z
#     --skip-claude          Don't install/update Claude Code
#     --no-chsh              Don't change the login shell to zsh
#     --update               Also `git pull` Oh My Zsh and zsh-autosuggestions
#     --git-name <name>      git user.name  (default: keep existing, else "Haoran Wan")
#     --git-email <email>    git user.email (default: keep existing, else haoran.w@princeton.edu)
#     --skip-git             Don't touch ~/.gitconfig
#     --ssh-key <path>       Key file to create/use (default: ~/.ssh/id_ed25519)
#     --skip-ssh             Don't generate a key or touch ~/.ssh/known_hosts
#     -h, --help             Show this help
#
#  References:
#     https://github.com/ohmyzsh/ohmyzsh#basic-installation
#     https://github.com/zsh-users/zsh-autosuggestions/blob/master/INSTALL.md
#     https://docs.claude.com/en/docs/claude-code/setup
#     https://docs.github.com/en/authentication/connecting-to-github-with-ssh
#     https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints
# =============================================================================

# Re-exec under bash if someone runs us with `sh setup-shell.sh`.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# ------------------------------------------------------------------ config --
OMZ_INSTALLER_URL="https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh"
AUTOSUGGESTIONS_REPO="https://github.com/zsh-users/zsh-autosuggestions"
CLAUDE_INSTALLER_URL="https://claude.ai/install.sh"
PLUGIN_NAME="zsh-autosuggestions"

SKIP_CLAUDE=0
DO_CHSH=1
DO_UPDATE=0
CLAUDE_CHANNEL=""     # empty => whatever the official installer defaults to

# Host-side git identity (commits made from the host; the container keeps its own
# /root/.gitconfig, see README.md "Identity and push auth"). Only applied when
# ~/.gitconfig has no value yet, unless passed explicitly on the command line.
GIT_NAME_DEFAULT="Haoran Wan"
GIT_EMAIL_DEFAULT="haoran.w@princeton.edu"
GIT_NAME=""           # set by --git-name  => forced
GIT_EMAIL=""          # set by --git-email => forced
SKIP_GIT=0
SKIP_SSH=0
SSH_KEY="$HOME/.ssh/id_ed25519"
# github.com's published ed25519 host key (fingerprint
# SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU). Pinned here so the first
# `git push` never hits an interactive "authenticity of host" prompt and never
# trusts a key that merely arrived over the wire.
GITHUB_HOST_KEY="github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl"
GITHUB_AUTH_OK=""     # filled in by setup_ssh_key: yes | no | skipped

ZSHRC="$HOME/.zshrc"
OMZ_DIR="${ZSH:-$HOME/.oh-my-zsh}"
OMZ_CUSTOM_DIR="${ZSH_CUSTOM:-$OMZ_DIR/custom}"
PLUGIN_DIR="$OMZ_CUSTOM_DIR/plugins/$PLUGIN_NAME"

BACKUP_MADE=0
BACKUP_PATH=""
CHANGED_ZSHRC=0
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
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/setup-shell.XXXXXX")

_tmpseq=0
mktempfile() {
  _tmpseq=$((_tmpseq + 1))
  local f="$WORKDIR/f$$.$_tmpseq"
  : > "$f"
  printf '%s\n' "$f"
}

usage() { sed -n '2,/^# ===\+$/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; }

# ------------------------------------------------------------------- args ---
while [ $# -gt 0 ]; do
  case "$1" in
    --claude-channel)
      [ $# -ge 2 ] || die "--claude-channel needs a value (stable|latest|X.Y.Z)"
      CLAUDE_CHANNEL="$2"; shift 2 ;;
    --claude-channel=*) CLAUDE_CHANNEL="${1#*=}"; shift ;;
    --skip-claude)      SKIP_CLAUDE=1; shift ;;
    --no-chsh)          DO_CHSH=0; shift ;;
    --update)           DO_UPDATE=1; shift ;;
    --git-name)
      [ $# -ge 2 ] || die "--git-name needs a value"
      GIT_NAME="$2"; shift 2 ;;
    --git-name=*)       GIT_NAME="${1#*=}"; shift ;;
    --git-email)
      [ $# -ge 2 ] || die "--git-email needs a value"
      GIT_EMAIL="$2"; shift 2 ;;
    --git-email=*)      GIT_EMAIL="${1#*=}"; shift ;;
    --skip-git)         SKIP_GIT=1; shift ;;
    --ssh-key)
      [ $# -ge 2 ] || die "--ssh-key needs a path"
      SSH_KEY="$2"; shift 2 ;;
    --ssh-key=*)        SSH_KEY="${1#*=}"; shift ;;
    --skip-ssh)         SKIP_SSH=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)                  die "Unknown option: $1  (try --help)" ;;
  esac
done

if [ -n "$CLAUDE_CHANNEL" ] &&
   ! [[ "$CLAUDE_CHANNEL" =~ ^(stable|latest|[0-9]+\.[0-9]+\.[0-9]+(-[^[:space:]]+)?)$ ]]; then
  die "--claude-channel must be 'stable', 'latest' or a version like 2.1.263"
fi

# ------------------------------------------------------------ preflight -----
# The Claude installer (and Oh My Zsh) install into $HOME. Under `sudo` $HOME
# is usually root's, so everything would land in the wrong place.
if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
  die "Don't run this script with sudo. Run it as your normal user; it will call sudo only for apt-get and chsh."
fi

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
elif command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

have_root() { [ "$(id -u)" -eq 0 ] || [ -n "$SUDO" ]; }

CURRENT_USER="${USER:-$(id -un)}"

# ------------------------------------------------------- 1. base packages ---
install_packages() {
  step "Base packages (zsh, git, curl)"
  local p missing=()
  for p in zsh git curl; do
    command -v "$p" >/dev/null 2>&1 || missing+=("$p")
  done

  if [ "${#missing[@]}" -eq 0 ]; then
    ok "zsh, git and curl are already installed"
    return 0
  fi

  info "Missing: ${missing[*]}"
  if ! command -v apt-get >/dev/null 2>&1; then
    die "apt-get not found. Install ${missing[*]} with your package manager and re-run."
  fi
  if ! have_root; then
    die "Need root (sudo) to install ${missing[*]}."
  fi

  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get update -qq
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
  ok "Installed: ${missing[*]}"
}

# ---------------------------------------------------------- 2. Oh My Zsh ----
install_omz() {
  step "Oh My Zsh"

  if [ -d "$OMZ_DIR/.git" ]; then
    ok "Already installed at $OMZ_DIR"
    if [ "$DO_UPDATE" -eq 1 ]; then
      info "Updating..."
      if git -C "$OMZ_DIR" pull --ff-only --quiet; then
        ok "Updated"
      else
        warn "git pull failed in $OMZ_DIR (leaving it as-is)"
      fi
    fi
    return 0
  fi

  if [ -e "$OMZ_DIR" ]; then
    die "$OMZ_DIR exists but is not a git checkout. Move it aside and re-run."
  fi

  local tmp
  tmp=$(mktempfile)
  info "Downloading $OMZ_INSTALLER_URL"
  curl -fsSL "$OMZ_INSTALLER_URL" -o "$tmp" || die "Failed to download the Oh My Zsh installer."

  # --unattended  : no `chsh` prompt, don't launch zsh when finished.
  # KEEP_ZSHRC=yes: never clobber an existing ~/.zshrc (the installer still
  #                 writes its template when ~/.zshrc does not exist yet).
  RUNZSH=no CHSH=no KEEP_ZSHRC=yes sh "$tmp" --unattended </dev/null \
    || die "The Oh My Zsh installer failed."

  [ -d "$OMZ_DIR" ] || die "Oh My Zsh installer finished but $OMZ_DIR is missing."
  ok "Installed at $OMZ_DIR"
}

# ------------------------------------------------- 3. zsh-autosuggestions ---
install_autosuggestions() {
  step "zsh-autosuggestions plugin"

  if [ -d "$PLUGIN_DIR/.git" ]; then
    ok "Already cloned at $PLUGIN_DIR"
    if [ "$DO_UPDATE" -eq 1 ]; then
      info "Updating..."
      if git -C "$PLUGIN_DIR" pull --ff-only --quiet; then
        ok "Updated"
      else
        warn "git pull failed in $PLUGIN_DIR (leaving it as-is)"
      fi
    fi
    return 0
  fi

  if [ -e "$PLUGIN_DIR" ]; then
    die "$PLUGIN_DIR exists but is not a git checkout. Move it aside and re-run."
  fi

  mkdir -p "$(dirname "$PLUGIN_DIR")"
  info "Cloning $AUTOSUGGESTIONS_REPO"
  git clone --quiet "$AUTOSUGGESTIONS_REPO" "$PLUGIN_DIR" \
    || die "git clone of zsh-autosuggestions failed."
  ok "Cloned to $PLUGIN_DIR"
}

# --------------------------------------------------------- 4. ~/.zshrc ------
backup_zshrc() {
  [ "$BACKUP_MADE" -eq 1 ] && return 0
  if [ -f "$ZSHRC" ]; then
    BACKUP_PATH="$ZSHRC.bak.$(date +%Y%m%d-%H%M%S)"
    cp -p "$ZSHRC" "$BACKUP_PATH"
    info "Backed up $ZSHRC -> $BACKUP_PATH"
  fi
  BACKUP_MADE=1
}

# Replace $ZSHRC with the contents of $1, keeping the original permissions.
replace_zshrc() {
  local new="$1"
  if [ -f "$ZSHRC" ]; then
    cat "$new" > "$ZSHRC"          # preserves mode/ownership/inode
  else
    cp "$new" "$ZSHRC"
  fi
  CHANGED_ZSHRC=1
}

# Does ~/.zshrc actually source Oh My Zsh?
zshrc_sources_omz() {
  grep -qE '^[[:space:]]*(source|\.)[[:space:]]+.*oh-my-zsh\.sh' "$ZSHRC"
}

# exit 0 = plugin present, 1 = absent, 2 = no plugins=( ... ) array found
zshrc_plugin_state() {
  awk -v want="$PLUGIN_NAME" '
    { line = $0; sub(/[[:space:]]*#.*$/, "", line) }
    state == 0 && line ~ /^[[:space:]]*plugins=\(/ {
      found = 1; blk = line
      state = (index(line, ")") > 0) ? 2 : 1
      next
    }
    state == 1 {
      blk = blk " " line
      if (line ~ /^[[:space:]]*\)/) state = 2
      next
    }
    END {
      if (!found) exit 2
      sub(/^[^(]*\(/, "", blk)
      sub(/\).*$/, "", blk)
      n = split(blk, a, /[ \t]+/)
      for (i = 1; i <= n; i++) if (a[i] == want) exit 0
      exit 1
    }
  ' "$ZSHRC"
}

# Insert $PLUGIN_NAME into the existing plugins=( ... ) array.
zshrc_add_plugin() {
  local tmp
  tmp=$(mktempfile)
  awk -v want="$PLUGIN_NAME" '
    BEGIN { state = 0; done = 0; indent = "  " }
    state == 0 && /^[[:space:]]*plugins=\(/ {
      if (index($0, ")") > 0) {
        if ($0 ~ /plugins=\([[:space:]]*\)/) sub(/\([[:space:]]*\)/, "(" want ")")
        else                                 sub(/\)/, " " want ")")
        print; state = 2; done = 1; next
      }
      print; state = 1; next
    }
    state == 1 {
      if ($0 ~ /^[[:space:]]*\)/) { print indent want; print; state = 2; done = 1; next }
      if ($0 ~ /[^[:space:]]/) { match($0, /^[[:space:]]*/); indent = substr($0, 1, RLENGTH) }
      print; next
    }
    { print }
    END { if (!done) exit 1 }
  ' "$ZSHRC" > "$tmp" || die "Failed to rewrite the plugins=(...) array in $ZSHRC"

  [ -s "$tmp" ] || die "Refusing to write an empty $ZSHRC"
  backup_zshrc
  replace_zshrc "$tmp"
  ok "Added '$PLUGIN_NAME' to plugins=(...) in $ZSHRC"
}

append_omz_bootstrap() {
  backup_zshrc
  cat >> "$ZSHRC" <<'BLOCK'

# >>> setup-shell.sh: Oh My Zsh bootstrap >>>
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="robbyrussell"
plugins=(git zsh-autosuggestions)
source "$ZSH/oh-my-zsh.sh"
# <<< setup-shell.sh: Oh My Zsh bootstrap <<<
BLOCK
  CHANGED_ZSHRC=1
  ok "Appended an Oh My Zsh bootstrap block to $ZSHRC"
}

ensure_local_bin_on_path() {
  # Claude Code installs to ~/.local/bin; make sure zsh can find it.
  # shellcheck disable=SC2088  # the tildes below are literal text for the user
  if grep -v '^[[:space:]]*#' "$ZSHRC" | grep -q '\.local/bin'; then
    ok "~/.local/bin is already on PATH in $ZSHRC"
    return 0
  fi
  backup_zshrc
  # shellcheck disable=SC2016  # $HOME must stay literal - zsh expands it, not us
  printf '\n# Claude Code and other user-local tools live here\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$ZSHRC"
  CHANGED_ZSHRC=1
  ok "Added ~/.local/bin to PATH in $ZSHRC"
}

configure_zshrc() {
  step "Configuring $ZSHRC"

  if [ ! -f "$ZSHRC" ]; then
    if [ -f "$OMZ_DIR/templates/zshrc.zsh-template" ]; then
      cp "$OMZ_DIR/templates/zshrc.zsh-template" "$ZSHRC"
      info "Created $ZSHRC from the Oh My Zsh template"
    else
      : > "$ZSHRC"
      info "Created an empty $ZSHRC"
    fi
    CHANGED_ZSHRC=1
    BACKUP_MADE=1   # nothing existed to back up
  fi

  if ! zshrc_sources_omz; then
    warn "$ZSHRC does not source oh-my-zsh.sh; appending a bootstrap block."
    append_omz_bootstrap
  else
    local state=0
    zshrc_plugin_state || state=$?
    case "$state" in
      0) ok "'$PLUGIN_NAME' is already in plugins=(...)" ;;
      1) zshrc_add_plugin ;;
      2) warn "No plugins=(...) array found; appending one before the oh-my-zsh.sh source line is not safe automatically."
         die "Add 'plugins=($PLUGIN_NAME)' above 'source \$ZSH/oh-my-zsh.sh' in $ZSHRC and re-run." ;;
    esac
  fi

  ensure_local_bin_on_path
}

# ------------------------------------------------------ 5. default shell ----
set_login_shell() {
  step "Default login shell"
  if [ "$DO_CHSH" -eq 0 ]; then
    info "Skipped (--no-chsh)"
    return 0
  fi

  local zsh_path current
  zsh_path=$(command -v zsh) || die "zsh not found after installation."
  current=$(getent passwd "$CURRENT_USER" | cut -d: -f7 || true)

  if [ "$current" = "$zsh_path" ]; then
    ok "Login shell is already $zsh_path"
    return 0
  fi

  if ! grep -qxF "$zsh_path" /etc/shells 2>/dev/null; then
    if have_root; then
      printf '%s\n' "$zsh_path" | $SUDO tee -a /etc/shells >/dev/null
      info "Registered $zsh_path in /etc/shells"
    else
      warn "$zsh_path is not in /etc/shells and we can't add it; chsh may refuse."
    fi
  fi

  if have_root && $SUDO chsh -s "$zsh_path" "$CURRENT_USER"; then
    ok "Login shell changed to $zsh_path (takes effect on next login)"
  elif chsh -s "$zsh_path" 2>/dev/null; then
    ok "Login shell changed to $zsh_path (takes effect on next login)"
  else
    warn "Could not change the login shell. Run: chsh -s $zsh_path"
  fi
}

# ------------------------------------------------------- 6. Claude Code -----
install_claude() {
  step "Claude Code"
  if [ "$SKIP_CLAUDE" -eq 1 ]; then
    info "Skipped (--skip-claude)"
    return 0
  fi

  local before=""
  if command -v claude >/dev/null 2>&1; then
    before=$(claude --version 2>/dev/null | head -1 || true)
    info "Currently installed: ${before:-unknown}"
  fi

  local tmp
  tmp=$(mktempfile)
  info "Downloading $CLAUDE_INSTALLER_URL"
  curl -fsSL "$CLAUDE_INSTALLER_URL" -o "$tmp" || die "Failed to download the Claude Code installer."

  # The official installer downloads the newest release, verifies its SHA-256
  # against the signed manifest, and installs into ~/.local/. Re-running it is
  # how you upgrade. It must NOT run under sudo.
  if [ -n "$CLAUDE_CHANNEL" ]; then
    bash "$tmp" "$CLAUDE_CHANNEL" </dev/null || die "The Claude Code installer failed."
  else
    bash "$tmp" </dev/null || die "The Claude Code installer failed."
  fi

  local bin="$HOME/.local/bin/claude"
  if [ -x "$bin" ]; then
    ok "claude $("$bin" --version 2>/dev/null | head -1) at $bin"
  else
    warn "Installer finished but $bin was not found."
  fi
}

# ------------------------------------------------- 7. git identity (host) ---
# Sets one global key if it is unset, or unconditionally when the value was given
# on the command line. Existing values are never silently replaced.
set_git_global() {           # set_git_global <key> <wanted> <forced 0|1>
  local key=$1 wanted=$2 forced=$3 current
  current=$(git config --global --get "$key" 2>/dev/null || true)
  if [ "$current" = "$wanted" ]; then
    ok "$key is already '$current'"
  elif [ -n "$current" ] && [ "$forced" -eq 0 ]; then
    ok "$key is already '$current' (kept; pass --${key#user.}=... to change it)"
  else
    git config --global "$key" "$wanted"
    if [ -n "$current" ]; then
      ok "$key: '$current' -> '$wanted'"
    else
      ok "$key set to '$wanted'"
    fi
  fi
}

configure_git_identity() {
  step "Git identity (~/.gitconfig)"
  if [ "$SKIP_GIT" -eq 1 ]; then
    info "Skipped (--skip-git)"
    return 0
  fi
  command -v git >/dev/null 2>&1 || die "git is not installed (install_packages should have handled this)."

  local forced_name=0 forced_email=0
  [ -n "$GIT_NAME" ]  && forced_name=1  || GIT_NAME="$GIT_NAME_DEFAULT"
  [ -n "$GIT_EMAIL" ] && forced_email=1 || GIT_EMAIL="$GIT_EMAIL_DEFAULT"
  set_git_global user.name  "$GIT_NAME"  "$forced_name"
  set_git_global user.email "$GIT_EMAIL" "$forced_email"
}

# ---------------------------------------------------- 8. SSH key for GitHub --
setup_ssh_key() {
  step "SSH key for GitHub ($SSH_KEY)"
  if [ "$SKIP_SSH" -eq 1 ]; then
    info "Skipped (--skip-ssh)"
    GITHUB_AUTH_OK="skipped"
    return 0
  fi
  command -v ssh-keygen >/dev/null 2>&1 || die "ssh-keygen not found; install openssh-client."

  local ssh_dir
  ssh_dir=$(dirname "$SSH_KEY")
  mkdir -p "$ssh_dir"
  chmod 700 "$ssh_dir"

  local comment="${GIT_EMAIL:-$GIT_EMAIL_DEFAULT}@$(hostname -s 2>/dev/null || hostname)"
  if [ -f "$SSH_KEY" ]; then
    ok "Key already exists: $SSH_KEY"
    [ -f "$SSH_KEY.pub" ] || { ssh-keygen -y -f "$SSH_KEY" > "$SSH_KEY.pub"; info "Regenerated missing $SSH_KEY.pub"; }
  else
    # No passphrase: this is an unattended dev box and the key only grants push
    # access to the fork. Use a passphrase + ssh-agent if that is not acceptable.
    ssh-keygen -q -t ed25519 -f "$SSH_KEY" -N "" -C "$comment" </dev/null
    ok "Generated ed25519 key $SSH_KEY (comment: $comment)"
  fi
  chmod 600 "$SSH_KEY"
  chmod 644 "$SSH_KEY.pub"

  # Pin github.com's host key. ssh-keygen -F understands hashed known_hosts entries.
  local kh="$ssh_dir/known_hosts"
  touch "$kh"; chmod 600 "$kh"
  if ssh-keygen -F github.com -f "$kh" >/dev/null 2>&1; then
    ok "github.com is already in $kh"
  else
    printf '%s\n' "$GITHUB_HOST_KEY" >> "$kh"
    ok "Pinned github.com's ed25519 host key in $kh"
  fi

  info "Public key (add it at https://github.com/settings/ssh/new if GitHub does not know it yet):"
  printf '\n        %s\n\n' "$(cat "$SSH_KEY.pub")"

  # Non-interactive auth probe. GitHub closes the session with exit 1 on success.
  local out
  out=$(ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=10 -o IdentitiesOnly=yes \
            -T git@github.com </dev/null 2>&1 || true)
  if printf '%s' "$out" | grep -q "successfully authenticated"; then
    GITHUB_AUTH_OK="yes"
    ok "GitHub accepts this key: $(printf '%s' "$out" | head -1)"
  else
    GITHUB_AUTH_OK="no"
    warn "GitHub did not accept the key yet ($(printf '%s' "$out" | tail -1 | cut -c1-80)). Add the public key above, then: ssh -T git@github.com"
  fi
}

# ---------------------------------------------------------- 9. verify -------
verify() {
  step "Verifying"

  if [ "$SKIP_GIT" -eq 0 ]; then
    local n e
    n=$(git config --global --get user.name 2>/dev/null || true)
    e=$(git config --global --get user.email 2>/dev/null || true)
    if [ -n "$n" ] && [ -n "$e" ]; then
      ok "git commits from the host will be authored as '$n <$e>'"
    else
      warn "git user.name/user.email are still unset; host-side commits will fail with 'Author identity unknown'."
    fi
  fi

  local runner=(zsh -i -c)
  command -v timeout >/dev/null 2>&1 && runner=(timeout 60 zsh -i -c)

  if DISABLE_AUTO_UPDATE=true DISABLE_UPDATE_PROMPT=true \
     "${runner[@]}" '(( $+functions[_zsh_autosuggest_start] ))' </dev/null >/dev/null 2>&1; then
    ok "zsh-autosuggestions loads in an interactive zsh"
  else
    warn "Could not confirm zsh-autosuggestions loads. Open a new shell and check manually."
  fi

  if [ "$SKIP_CLAUDE" -eq 0 ]; then
    if DISABLE_AUTO_UPDATE=true DISABLE_UPDATE_PROMPT=true \
       "${runner[@]}" 'command -v claude >/dev/null' </dev/null >/dev/null 2>&1; then
      ok "'claude' is on PATH in an interactive zsh"
    else
      warn "'claude' is not on PATH in an interactive zsh yet."
    fi
  fi
}

# ------------------------------------------------------------- summary ------
summary() {
  step "Done"
  info "zsh:               $(zsh --version 2>/dev/null || echo 'n/a')"
  info "Oh My Zsh:         $OMZ_DIR"
  info "autosuggestions:   $PLUGIN_DIR"
  if [ "$SKIP_CLAUDE" -eq 0 ] && [ -x "$HOME/.local/bin/claude" ]; then
    info "Claude Code:       $("$HOME/.local/bin/claude" --version 2>/dev/null | head -1)"
  fi
  if [ "$SKIP_GIT" -eq 0 ]; then
    info "git identity:      $(git config --global --get user.name 2>/dev/null || echo '?') <$(git config --global --get user.email 2>/dev/null || echo '?')>"
  fi
  if [ "$SKIP_SSH" -eq 0 ]; then
    info "SSH key:           $SSH_KEY (GitHub auth: ${GITHUB_AUTH_OK:-unknown})"
  fi
  if [ -n "$BACKUP_PATH" ]; then
    # shellcheck disable=SC2088  # literal text
    info "~/.zshrc backup:   $BACKUP_PATH"
  fi
  if [ "${GITHUB_AUTH_OK:-}" = "no" ]; then
    printf '\n    %sPaste %s.pub at https://github.com/settings/ssh/new, then re-run this script or: ssh -T git@github.com%s\n' "$C_Y" "$SSH_KEY" "$C_0"
  fi
  # Host-side pushes need an SSH remote; the fork is cloned over HTTPS by default.
  local repo=/lambda/nfs/MLSys-Learn/sglang
  if [ "$SKIP_SSH" -eq 0 ] && [ -d "$repo/.git" ] && git -C "$repo" remote get-url origin 2>/dev/null | grep -q '^https://github.com/'; then
    printf '\n    %sTo push from the host over SSH: git -C %s remote set-url origin git@github.com:WanHaoRan/sglang.git%s\n' "$C_B" "$repo" "$C_0"
  fi
  if [ "$CHANGED_ZSHRC" -eq 1 ]; then
    printf '\n    %sStart a new shell (or run: exec zsh) to pick up the changes.%s\n' "$C_B" "$C_0"
  else
    printf '\n    %sNothing to change - everything was already in place.%s\n' "$C_B" "$C_0"
  fi
}

# ---------------------------------------------------------------- main ------
main() {
  printf '%s%s%s\n' "$C_B" "setup-shell.sh - zsh + Oh My Zsh + zsh-autosuggestions + Claude Code + git identity + GitHub SSH key" "$C_0"
  install_packages
  install_omz
  install_autosuggestions
  configure_zshrc
  set_login_shell
  install_claude
  configure_git_identity
  setup_ssh_key
  verify
  summary
}

main "$@"
