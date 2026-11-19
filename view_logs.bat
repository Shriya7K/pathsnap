@echo off
cd /d "%~dp0"
echo ====================================================================
echo   PATHSNAP LIVE SERVER ACTIVITY MONITOR (CMD)
echo   Listening to host server at http://127.0.0.1:8000
echo ====================================================================
"%LOCALAPPDATA%\Programs\Python\Python314\python.exe" -u -c "import urllib.request, json, time; print('Connected. Streaming inbound client matches in real time... (Press Ctrl+C to stop)\n'); last_ts = 0; [(print(f'[{time.strftime(\"%%H:%%M:%%S\", time.localtime(latest[\"timestamp\"]))}] CLIENT {latest.get(\"client_ip\")}: Trip {latest.get(\"trip_id\")} ({latest.get(\"point_count\")} pts | {round(latest.get(\"total_distance_m\",0)/1000, 2)} km | Acc: {latest.get(\"ground_truth_accuracy\")}%%)', setattr(sys.modules[__name__], 'last_ts', latest['timestamp'])) for _ in iter(int, 1) if (stats := json.loads(urllib.request.urlopen('http://127.0.0.1:8000/api/server-stats', timeout=2).read().decode())) and (latest := stats.get('latest_activity')) and latest.get('timestamp', 0) > last_ts and time.sleep(1.5) is None or time.sleep(1.5) is None]"
pause
