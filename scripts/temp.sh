ps -C codex -o pid,ppid,stat,cmd
pkill -TERM -u root -x codex 2>/dev/null || true
sleep 2
pkill -KILL -u root -x codex 2>/dev/null || true
ps -C codex -o pid,ppid,stat,cmd



if [ -d /tmp/codex-daemon-0 ]; then
  mv /tmp/codex-daemon-0 "/tmp/codex-daemon-0.stale.$(date +%s)"
fi
codex