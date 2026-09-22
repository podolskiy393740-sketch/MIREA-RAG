#!/usr/bin/env bash
# Деплой веб-версии МИРЭА RAG на VPS.
# Запускать от root или через sudo.
# Использование: sudo bash deploy/install_web.sh
set -euo pipefail

# ── Переменные ────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_NAME="mirea-web"
VPS_USER="${SUDO_USER:-$(logname 2>/dev/null || echo "ubuntu")}"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
LOGROTATE_FILE="/etc/logrotate.d/${SERVICE_NAME}"
LOG_DIR="/var/log/mirea"

echo ">>> PROJECT_DIR  = $PROJECT_DIR"
echo ">>> VPS_USER     = $VPS_USER"

# ── Лог-директория ────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
chown "$VPS_USER":"$VPS_USER" "$LOG_DIR"

# ── systemd-юнит ──────────────────────────────────────────────────────────────
sed \
  -e "s|YOUR_VPS_USER|${VPS_USER}|g" \
  -e "s|YOUR_PROJECT_DIR|${PROJECT_DIR}|g" \
  "${PROJECT_DIR}/deploy/mirea-web.service" \
  > "$SERVICE_FILE"

echo ">>> Установлен юнит: $SERVICE_FILE"

# ── logrotate ─────────────────────────────────────────────────────────────────
cp "${PROJECT_DIR}/deploy/mirea-web.logrotate" "$LOGROTATE_FILE"
echo ">>> Установлен logrotate: $LOGROTATE_FILE"

# ── Запуск ────────────────────────────────────────────────────────────────────
systemctl daemon-reload
systemctl enable --now "$SERVICE_NAME"
systemctl status "$SERVICE_NAME" --no-pager -l
