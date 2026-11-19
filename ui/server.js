/**
 * PathSnap Server Monitor - Dedicated Host Application
 */

// Application State
const serverState = {
  map: null,
  rawGpsLayer: null,
  snappedRouteLayer: null,
  lastSyncedTimestamp: 0,
  prevActiveClientsCount: 0,
};

// DOM Elements
const statConnectedClients = document.getElementById("statConnectedClients");
const statProcessedMatches = document.getElementById("statProcessedMatches");
const clientCountBadge = document.getElementById("clientCountBadge");
const clientsListBox = document.getElementById("clientsListBox");
const activityFeedBox = document.getElementById("activityFeedBox");
const metricsPanel = document.getElementById("metricsPanel");
const valActiveClient = document.getElementById("valActiveClient");
const valPoints = document.getElementById("valPoints");
const valDistance = document.getElementById("valDistance");
const valConfidence = document.getElementById("valConfidence");
const valAccuracy = document.getElementById("valAccuracy");
const accuracyCard = document.getElementById("accuracyCard");
const logTerminal = document.getElementById("logTerminal");
const btnClearLogs = document.getElementById("btnClearLogs");
const btnResetServer = document.getElementById("btnResetServer");
const overlayTripText = document.getElementById("overlayTripText");

// Logging Utility
function logMessage(msg, type = "info") {
  if (!logTerminal) return;
  const timeStr = new Date().toLocaleTimeString();
  const line = document.createElement("div");
  line.className = `log-line ${type}`;
  line.style.color = "#ffffff";
  line.innerHTML = `<span style="color: #38bdf8; font-weight: 500;">[${timeStr}]</span> <span style="color: #ffffff !important;">${msg}</span>`;
  logTerminal.appendChild(line);
  logTerminal.scrollTop = logTerminal.scrollHeight;
}

// 1. Initialize Map
function initMap() {
  serverState.map = L.map("map", {
    zoomControl: true,
    attributionControl: true,
  }).setView([51.505, -0.09], 13);

  // Esri World Street Map: Clean, reliable, no watermark
  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    {
      maxZoom: 19,
      attribution: "Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ",
    }
  ).addTo(serverState.map);

  serverState.rawGpsLayer = L.featureGroup().addTo(serverState.map);
  serverState.snappedRouteLayer = L.featureGroup().addTo(serverState.map);

  // Force Leaflet to recalculate container dimensions
  setTimeout(() => {
    if (serverState.map) serverState.map.invalidateSize();
  }, 100);
  setTimeout(() => {
    if (serverState.map) serverState.map.invalidateSize();
  }, 500);
}

// 2. Poll Server Telemetry & Connected Clients
async function pollServerTelemetry() {
  try {
    const res = await fetch("/api/server-stats");
    if (!res.ok) return;
    const stats = await res.json();

    const activeCount = stats.active_client_count || 0;
    if (statConnectedClients) statConnectedClients.textContent = activeCount;
    if (statProcessedMatches) statProcessedMatches.textContent = (stats.request_history || []).length;
    if (clientCountBadge) {
      clientCountBadge.textContent = `● ${activeCount} Active Client${activeCount === 1 ? "" : "s"}`;
      clientCountBadge.style.background = activeCount > 0 ? "#ecfdf5" : "#f1f5f9";
      clientCountBadge.style.color = activeCount > 0 ? "#047857" : "#64748b";
      clientCountBadge.style.borderColor = activeCount > 0 ? "#10b981" : "#cbd5e1";
    }

    // Notify when client connectivity changes
    if (serverState.prevActiveClientsCount > 0 && activeCount === 0) {
      logMessage("⚠️ Client went offline. Last matched route remains on map for evaluation review.", "action");
    } else if (serverState.prevActiveClientsCount === 0 && activeCount > 0) {
      logMessage(`🟢 ${activeCount} Client(s) connected! Ready to receive trajectories.`, "success");
    }
    serverState.prevActiveClientsCount = activeCount;

    // Render Connected Clients List
    if (clientsListBox) {
      const activeClients = (stats.clients || []).filter((c) => c.is_active);
      if (activeClients.length === 0) {
        clientsListBox.innerHTML = '<div class="client-item idle" style="padding: 10px; text-align: center; color: #64748b; font-size: 0.8rem;">No clients connected. Waiting for Laptop 2...</div>';
      } else {
        clientsListBox.innerHTML = "";
        activeClients.forEach((c) => {
          const item = document.createElement("div");
          item.className = "client-card active";
          item.innerHTML = `
            <div>
              <div class="client-ip-text">📱 ${c.ip}</div>
              <div style="font-size: 0.7rem; color: #64748b;">Heartbeat: ${c.last_seen_sec_ago}s ago &bull; Action: ${c.last_action}</div>
            </div>
            <div style="text-align: right;">
              <span style="font-size: 0.72rem; font-weight: 700; color: #10b981;">● ONLINE</span>
              <div style="font-size: 0.7rem; color: #475569;">${c.requests_count} matches</div>
            </div>
          `;
          clientsListBox.appendChild(item);
        });
      }
    }

    // Check Latest Activity to mirror map
    const latest = stats.latest_activity;
    if (latest && latest.timestamp && latest.timestamp > serverState.lastSyncedTimestamp) {
      serverState.lastSyncedTimestamp = latest.timestamp;
      renderMirroredMatch(latest);
    }
  } catch (err) {
    // Network poll jitter
  }
}

// 3. Render Mirrored Match on Server Map
function renderMirroredMatch(data) {
  logMessage(`📡 Inbound match received from Client ${data.client_ip}: Trip ${data.trip_id} (${data.point_count} pts)`, "system");

  // Update Inbound Activity Feed Box
  if (activityFeedBox) {
    const timeStr = new Date(data.timestamp * 1000).toLocaleTimeString();
    const line = document.createElement("div");
    line.className = "activity-feed-line";
    const accStr = data.ground_truth_accuracy !== null && data.ground_truth_accuracy !== undefined
      ? ` &bull; <span class="activity-acc">${data.ground_truth_accuracy}% Acc</span>`
      : "";
    line.innerHTML = `
      <span class="activity-time">[${timeStr}]</span> <span class="activity-ip">${data.client_ip}</span>: 
      Trip <b>${data.trip_id}</b> (${data.point_count} pts &bull; ${(data.total_distance_m / 1000.0).toFixed(2)} km${accStr})
    `;
    activityFeedBox.insertBefore(line, activityFeedBox.firstChild);
    while (activityFeedBox.children.length > 20) {
      activityFeedBox.removeChild(activityFeedBox.lastChild);
    }
  }

  // Update Metrics Panel
  if (metricsPanel) {
    metricsPanel.classList.remove("hidden");
    if (valActiveClient) valActiveClient.textContent = `Client: ${data.client_ip} (Trip ${data.trip_id})`;
    if (valPoints) valPoints.textContent = data.point_count.toLocaleString();
    if (valDistance) valDistance.textContent = (data.total_distance_m / 1000.0).toFixed(2) + " km";
    if (valConfidence) valConfidence.textContent = (data.mean_confidence * 100.0).toFixed(1) + "%";
    if (valAccuracy && accuracyCard) {
      if (data.ground_truth_accuracy !== null && data.ground_truth_accuracy !== undefined) {
        valAccuracy.textContent = data.ground_truth_accuracy + "%";
        accuracyCard.classList.remove("hidden");
      } else {
        accuracyCard.classList.add("hidden");
      }
    }
  }

  if (overlayTripText) {
    overlayTripText.textContent = `Client ${data.client_ip} • Trip ${data.trip_id} (${(data.total_distance_m / 1000.0).toFixed(2)} km)`;
  }

  // Draw on Map
  if (serverState.map) serverState.map.invalidateSize();
  serverState.rawGpsLayer.clearLayers();
  serverState.snappedRouteLayer.clearLayers();

  const allLatLngs = [];

  // Draw Raw GPS Points (Red)
  if (data.raw_coordinates && data.raw_coordinates.length > 0) {
    const rawCoords = data.raw_coordinates;
    const step = Math.max(1, Math.floor(rawCoords.length / 300));
    for (let i = 0; i < rawCoords.length; i += step) {
      const lat = rawCoords[i][0];
      const lon = rawCoords[i][1];
      if (isNaN(lat) || isNaN(lon)) continue;
      const marker = L.circleMarker([lat, lon], {
        radius: 3,
        color: "#ef4444",
        fillColor: "#ef4444",
        fillOpacity: 0.8,
        weight: 1,
      });
      serverState.rawGpsLayer.addLayer(marker);
      allLatLngs.push([lat, lon]);
    }
  }

  // Draw Snapped Route (Emerald Green)
  if (data.matched_coordinates && data.matched_coordinates.length > 0) {
    const validCoords = data.matched_coordinates.filter((c) => !isNaN(c[0]) && !isNaN(c[1]));
    if (validCoords.length > 0) {
      const snappedPolyline = L.polyline(validCoords, {
        color: "#10b981",
        weight: 5,
        opacity: 0.9,
        lineCap: "round",
        lineJoin: "round",
      }).addTo(serverState.snappedRouteLayer);

      const accInfo = data.ground_truth_accuracy ? `<br><b>Ground-Truth Acc:</b> ${data.ground_truth_accuracy}%` : "";
      snappedPolyline.bindPopup(`
        <div style="font-family: sans-serif; font-size: 0.8rem; line-height: 1.4;">
          <b style="color: #047857;">HMM Viterbi Matched Route</b><br>
          <b>Trip:</b> ${data.trip_id}<br>
          <b>Client:</b> ${data.client_ip}<br>
          <b>Total Points:</b> ${data.point_count}<br>
          <b>Distance:</b> ${(data.total_distance_m / 1000.0).toFixed(2)} km<br>
          <b>Execution Time:</b> ${data.execution_time_sec}s
          ${accInfo}
        </div>
      `);

      validCoords.forEach((c) => allLatLngs.push(c));
    }
  }

  if (allLatLngs.length > 0) {
    serverState.map.fitBounds(L.latLngBounds(allLatLngs), { padding: [40, 40] });
  }

  logMessage(`✅ Successfully synchronized and rendered Trip ${data.trip_id} on Server Map.`, "success");
}

// 4. Reset Server Button
if (btnResetServer) {
  btnResetServer.addEventListener("click", async () => {
    try {
      await fetch("/api/server-reset", { method: "POST" });
      serverState.rawGpsLayer.clearLayers();
      serverState.snappedRouteLayer.clearLayers();
      if (metricsPanel) metricsPanel.classList.add("hidden");
      if (overlayTripText) overlayTripText.textContent = "Waiting for Client Match Request";
      serverState.lastSyncedTimestamp = Date.now() / 1000;
      serverState.map.setView([51.505, -0.09], 13);
      logMessage("↺ Server monitor and live map reset to clean state.", "system");
    } catch (e) {
      logMessage("Failed to reset server state.", "error");
    }
  });
}

// Clear Logs
if (btnClearLogs) {
  btnClearLogs.addEventListener("click", () => {
    if (logTerminal) logTerminal.innerHTML = "";
  });
}

// App Initialization
document.addEventListener("DOMContentLoaded", () => {
  initMap();
  logMessage("🖥️ PathSnap Server Monitor initialized on host 10.20.202.73:8000.", "system");
  logMessage("Listening for client heartbeats and inbound match trajectories...", "info");
  
  // Start telemetry loop (every 1.5 seconds)
  pollServerTelemetry();
  setInterval(pollServerTelemetry, 1500);

  window.addEventListener("resize", () => {
    if (serverState.map) serverState.map.invalidateSize();
  });
});
