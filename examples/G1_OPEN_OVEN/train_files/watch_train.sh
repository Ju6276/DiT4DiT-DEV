#!/bin/bash
# 在 DSW / 本机挂着同一份 CPFS 时，用来判断 DLC 训练是否在推进。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOG_ROOT="${ROOT}/log"
INTERVAL="${1:-15}"

latest_dir="$(ls -dt "${LOG_ROOT}"/dit4dit_G1_OPEN_OVEN_* 2>/dev/null | head -1 || true)"
if [[ -z "${latest_dir}" ]]; then
  echo "未找到 OpenOven 训练日志目录: ${LOG_ROOT}/dit4dit_G1_OPEN_OVEN_*"
  exit 1
fi

log_file="${latest_dir}/train.log"
echo "Watching: ${log_file}"
echo "每 ${INTERVAL}s 刷新；看到 Step 且 mtime 在更新 = 仍在跑"
echo "=============================================="

prev_mtime=""
prev_steps=""
while true; do
  now="$(date '+%F %T')"
  if [[ ! -f "${log_file}" ]]; then
    echo "[${now}] 日志文件还不存在，等待..."
    sleep "${INTERVAL}"
    continue
  fi

  mtime="$(stat -c '%y' "${log_file}" 2>/dev/null | cut -d. -f1)"
  size="$(stat -c '%s' "${log_file}" 2>/dev/null)"
  last_step="$(grep -oE 'Step [0-9]+' "${log_file}" | tail -1 || true)"
  last_loss="$(grep 'Step ' "${log_file}" | tail -1 || true)"
  cfg_bs="$(grep -E 'Per device batch size' "${log_file}" | tail -1 || true)"

  status="UNKNOWN"
  if [[ -n "${last_step}" ]]; then
    if [[ "${mtime}" != "${prev_mtime}" || "${last_step}" != "${prev_steps}" ]]; then
      status="RUNNING (有新 Step / 日志在更新)"
    else
      status="STALE (已有 Step，但 ${INTERVAL}s 内无更新 — 可能卡住或刚保存完)"
    fi
  else
    if [[ "${mtime}" != "${prev_mtime}" ]]; then
      status="STARTING (配置/加载中，尚无 Step)"
    else
      status="STUCK? (无 Step 且日志不更新 — 看 DLC 控制台是否 OOM)"
    fi
  fi

  echo "[${now}] ${status}"
  echo "  mtime=${mtime}  size=${size}B  ${last_step:-Step ?}"
  [[ -n "${cfg_bs}" ]] && echo "  ${cfg_bs}"
  [[ -n "${last_loss}" ]] && echo "  ${last_loss}"
  echo "----------------------------------------------"

  prev_mtime="${mtime}"
  prev_steps="${last_step}"
  sleep "${INTERVAL}"
done
