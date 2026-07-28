#!/bin/bash
# Push the DIT4SIMPLE branch to the DiT4DiT-DEV GitHub repository.
#
# Usage:
#   bash push_github.sh                       # push DIT4SIMPLE
#   bash push_github.sh other-branch          # push another branch
#   bash push_github.sh --force               # force-with-lease push
#   bash push_github.sh --no-proxy            # push without proxy
#
# Environment overrides:
#   GITHUB_REMOTE       default: github
#   GITHUB_REMOTE_URL   default: https://github.com/Ju6276/DiT4DiT-DEV.git
#   HTTP_PROXY / HTTPS_PROXY   used when proxy is enabled

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

REMOTE="${GITHUB_REMOTE:-github}"
REMOTE_URL="${GITHUB_REMOTE_URL:-https://github.com/Ju6276/DiT4DiT-DEV.git}"
FORCE=0
USE_PROXY=1
BRANCH="DIT4SIMPLE"

usage() {
  sed -n '2,12p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -f|--force)
      FORCE=1
      shift
      ;;
    --no-proxy)
      USE_PROXY=0
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      if [[ "${BRANCH}" == "DIT4SIMPLE" ]]; then
        BRANCH="$1"
      else
        echo "Unexpected argument: $1" >&2
        usage
        exit 1
      fi
      shift
      ;;
  esac
done

CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${BRANCH}" ]]; then
  echo "Error: current branch is '${CURRENT_BRANCH:-detached HEAD}', expected '${BRANCH}'." >&2
  echo "Switch first with: git switch ${BRANCH}" >&2
  exit 1
fi

for excluded_dir in Cosmos-Predict2.5-2B datasets results; do
  if ! git check-ignore -q "${excluded_dir}"; then
    echo "Error: '${excluded_dir}/' is not excluded by .gitignore." >&2
    exit 1
  fi
done

enable_proxy() {
  if declare -F proxy_on >/dev/null 2>&1; then
    proxy_on
    return
  fi

  export http_proxy="${HTTP_PROXY:-http://172.16.3.158:3128}"
  export https_proxy="${HTTPS_PROXY:-http://172.16.3.158:3128}"
  export no_proxy="${NO_PROXY:-localhost,127.0.0.1,.aliyuncs.com,.internal}"
  echo "Proxy enabled: ${https_proxy}"
}

disable_proxy() {
  if declare -F proxy_off >/dev/null 2>&1; then
    proxy_off
    return
  fi

  unset http_proxy https_proxy no_proxy
  echo "Proxy disabled"
}

if [[ "${USE_PROXY}" -eq 1 ]]; then
  enable_proxy
else
  disable_proxy
fi

echo "Repository: ${PROJECT_ROOT}"
echo "Remote:     ${REMOTE}"
echo "Remote URL: ${REMOTE_URL}"
echo "Branch:     ${BRANCH}"
echo "Force push: $([[ "${FORCE}" -eq 1 ]] && echo yes || echo no)"
echo

git status --short --branch

if ! git diff-index --quiet HEAD -- 2>/dev/null || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  echo
  echo "Error: working tree has uncommitted changes. Commit them before pushing." >&2
  exit 1
fi

if git remote get-url "${REMOTE}" >/dev/null 2>&1; then
  configured_url="$(git remote get-url "${REMOTE}")"
  if [[ "${configured_url}" != "${REMOTE_URL}" ]]; then
    echo "Error: remote '${REMOTE}' points to '${configured_url}', expected '${REMOTE_URL}'." >&2
    exit 1
  fi
else
  git remote add "${REMOTE}" "${REMOTE_URL}"
fi

echo
read -r -p "Continue push to ${REMOTE}/${BRANCH}? [y/N] " confirm
if [[ ! "${confirm}" =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 0
fi

PUSH_ARGS=(-u "${REMOTE}" "${BRANCH}")
if [[ "${FORCE}" -eq 1 ]]; then
  PUSH_ARGS=(--force-with-lease "${PUSH_ARGS[@]}")
fi

git push "${PUSH_ARGS[@]}"

echo
echo "Done: ${REMOTE}/${BRANCH}"
