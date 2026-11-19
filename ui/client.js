/**
 * PathSnap Client Portal - Dedicated Client Application
 */

// Application State
const clientState = {
  map: null,
  rawGpsLayer: null,
  snappedRouteLayer: null,
  currentTripData: null,
  availableTrips: [],
};

// DOM Elements
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const sampleTripSelect = document.getElementById("sampleTripSelect");
const btnLoadSample = document.getElementById("btnLoadSample");
const btnMatchSample = document.getElementById("btnMatchSample");
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const btnUploadMatch = document.getElementById("btnUploadMatch");
const uploadTripId = document.getElementById("uploadTripId");
const manualCoords = document.getElementById("manualCoords");
const manualTripId = document.getElementById("manualTripId");
const btnManualMatch = document.getElementById("btnManualMatch");
const alertBox = document.getElementById("alertBox");
const alertMsg = document.getElementById("alertMsg");
const alertClose = document.getElementById("alertClose");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingText = document.getElementById("loadingText");
const metricsPanel = document.getElementById("metricsPanel");
const valPoints = document.getElementById("valPoints");
const valDistance = document.getElementById("valDistance");
const valConfidence = document.getElementById("valConfidence");
const valLatency = document.getElementById("valLatency");
const accuracyCard = document.getElementById("accuracyCard");
const valAccuracy = document.getElementById("valAccuracy");
const arcsTableBody = document.getElementById("arcsTableBody");
const arcCountBadge = document.getElementById("arcCountBadge");
const toggleRawGps = document.getElementById("toggleRawGps");
const toggleSnappedRoute = document.getElementById("toggleSnappedRoute");
const logTerminal = document.getElementById("logTerminal");
const btnClearLogs = document.getElementById("btnClearLogs");
const heartbeatBadge = document.getElementById("heartbeatBadge");

// Logging Utility
function logMessage(msg, type = "info") {
  if (!logTerminal) return;
  const timeStr = new Date().toLocaleTimeString();
  const line = document.createElement("div");
  line.className = `log-line ${type}`;
  line.textContent = `[${timeStr}] ${msg}`;
  logTerminal.appendChild(line);
  logTerminal.scrollTop = logTerminal.scrollHeight;
}

// 1. Initialize Leaflet Map
function initMap() {
  clientState.map = L.map("map", {
    zoomControl: true,
    attributionControl: true,
  }).setView([51.505, -0.09], 13);

  // Esri World Street Map (zero watermark)
  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    {
      maxZoom: 19,
      attribution: "Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ",
    }
  ).addTo(clientState.map);

  clientState.rawGpsLayer = L.featureGroup().addTo(clientState.map);
  clientState.snappedRouteLayer = L.featureGroup().addTo(clientState.map);

  // Force Leaflet to recalculate container dimensions
  setTimeout(() => {
    if (clientState.map) clientState.map.invalidateSize();
  }, 100);
  setTimeout(() => {
    if (clientState.map) clientState.map.invalidateSize();
  }, 500);
}

// 2. Tab Navigation
function setupTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));

      tab.classList.add("active");
      const targetPane = document.getElementById(tab.dataset.tab);
      if (targetPane) targetPane.classList.add("active");
    });
  });
}

// 3. Heartbeat Transmission to Host Server (Every 4 seconds)
function startHeartbeat() {
  const sendPing = async () => {
    try {
      const res = await fetch("/api/client-heartbeat");
      if (res.ok) {
        if (heartbeatBadge) {
          heartbeatBadge.textContent = "💓 Heartbeat: Connected (4s)";
          heartbeatBadge.style.color = "#047857";
          heartbeatBadge.style.background = "#ecfdf5";
          heartbeatBadge.style.borderColor = "#10b981";
        }
      }
    } catch (e) {
      if (heartbeatBadge) {
        heartbeatBadge.textContent = "💔 Disconnected from Host";
        heartbeatBadge.style.color = "#b91c1c";
        heartbeatBadge.style.background = "#fef2f2";
        heartbeatBadge.style.borderColor = "#ef4444";
      }
    }
  };

  sendPing();
  setInterval(sendPing, 4000);
}

// Unload Beacon on Tab Close / Refresh
window.addEventListener("beforeunload", () => {
  try {
    navigator.sendBeacon("/api/client-disconnect");
  } catch (e) {}
});

// 4. Fetch Sample Trips from Student Folder
async function loadSampleTrips() {
  try {
    const res = await fetch("/api/trips");
    if (!res.ok) throw new Error("Failed to fetch trips list");
    const data = await res.json();
    clientState.availableTrips = data.trips;

    sampleTripSelect.innerHTML = "";

    const studentTrips = data.trips.filter((t) => t.partition === "student_test");
    const trainTrips = data.trips.filter((t) => t.partition === "train");

    // Group 1: Student Test Trips
    if (studentTrips.length > 0) {
      const grpStudent = document.createElement("optgroup");
      grpStudent.label = `--- STUDENT TEST TRIPS (${studentTrips.length} TRIPS) ---`;
      studentTrips.forEach((t) => {
        const opt = document.createElement("option");
        opt.value = `${t.partition}/${t.trip_id}`;
        opt.textContent = `Student Test - Trip ${t.trip_id} (${t.point_count} pts)`;
        grpStudent.appendChild(opt);
      });
      sampleTripSelect.appendChild(grpStudent);
    }

    // Group 2: Training Trips
    if (trainTrips.length > 0) {
      const grpTrain = document.createElement("optgroup");
      grpTrain.label = `--- TRAINING TRIPS (${trainTrips.length} TRIPS) ---`;
      trainTrips.forEach((t) => {
        const opt = document.createElement("option");
        opt.value = `${t.partition}/${t.trip_id}`;
        opt.textContent = `Train - Trip ${t.trip_id} (${t.point_count} pts)`;
        grpTrain.appendChild(opt);
      });
      sampleTripSelect.appendChild(grpTrain);
    }

    logMessage(`Loaded ${data.total_trips || data.trips.length} trips from Student Folder.`, "info");

    // Default select Trip 00000007 if available, otherwise first student test trip
    const defaultTrip = studentTrips.find((t) => t.trip_id === "00000007") || studentTrips[0] || data.trips[0];
    if (defaultTrip) {
      sampleTripSelect.value = `${defaultTrip.partition}/${defaultTrip.trip_id}`;
      // Automatically load the default trip trajectory
      loadTripData(defaultTrip.partition, defaultTrip.trip_id);
    }
  } catch (err) {
    showAlert("Could not load sample trips: " + err.message);
    logMessage(`Failed to load trip catalogue: ${err.message}`, "error");
  }
}

// 5. Load Trip Coordinates
async function loadTripData(partition, tripId) {
  showLoading(`Loading GPS trajectory for Trip ${tripId}...`);
  try {
    const res = await fetch(`/api/trip/${partition}/${tripId}`);
    if (!res.ok) throw new Error(`Trip ${tripId} not found`);
    const data = await res.json();

    clientState.currentTripData = data;
    logMessage(`Loaded Trip ${tripId} (${data.point_count} GPS points, partition=${partition}).`, "info");

    renderRawGps(data.coordinates);
    btnMatchSample.disabled = false;
    btnMatchSample.focus();
    hideAlert();
  } catch (err) {
    showAlert(`Error loading trip ${tripId}: ${err.message}`);
    logMessage(`Trip load error: ${err.message}`, "error");
  } finally {
    hideLoading();
  }
}

// 6. Execute Map Matching
async function executeMapMatch(tripData) {
  showLoading(`Transmitting ${tripData.point_count} points to Host Server for HMM matching...`);
  try {
    const payload = {
      trip_id: tripData.trip_id,
      partition: tripData.partition,
      coordinates: tripData.coordinates,
    };

    const res = await fetch("/api/match", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const result = await res.json();
    if (!res.ok) throw new Error(result.detail || "Map matching failed");

    renderMatchResults(result);
    logMessage(`Match completed: ${result.point_count} pts snapped in ${result.execution_time_sec}s. Mean Conf: ${(result.mean_confidence*100).toFixed(1)}%`, "success");
    if (result.ground_truth_accuracy !== null && result.ground_truth_accuracy !== undefined) {
      logMessage(`Ground-Truth Sequence Accuracy: ${result.ground_truth_accuracy}%`, "system");
    }
  } catch (err) {
    showAlert(`Matching failed: ${err.message}`);
    logMessage(`Match error: ${err.message}`, "error");
  } finally {
    hideLoading();
  }
}

// 7. Render Raw GPS Points (Red)
function renderRawGps(coords) {
  if (clientState.map) clientState.map.invalidateSize();
  clientState.rawGpsLayer.clearLayers();
  clientState.snappedRouteLayer.clearLayers();

  if (!coords || coords.length === 0) return;

  const latLngs = [];
  const step = Math.max(1, Math.floor(coords.length / 400));

  for (let i = 0; i < coords.length; i += step) {
    const pt = coords[i];
    const lat = pt.latitude || pt.lat;
    const lon = pt.longitude || pt.lon;
    if (isNaN(lat) || isNaN(lon)) continue;

    const marker = L.circleMarker([lat, lon], {
      radius: 3,
      color: "#ef4444",
      fillColor: "#ef4444",
      fillOpacity: 0.8,
      weight: 1,
    });
    clientState.rawGpsLayer.addLayer(marker);
    latLngs.push([lat, lon]);
  }

  if (latLngs.length > 0) {
    clientState.map.fitBounds(L.latLngBounds(latLngs), { padding: [40, 40] });
  }
}

// 8. Render Snapped Route (Emerald Green)
function renderMatchResults(data) {
  if (clientState.map) clientState.map.invalidateSize();
  clientState.snappedRouteLayer.clearLayers();

  // Metrics panel
  metricsPanel.classList.remove("hidden");
  valPoints.textContent = data.point_count.toLocaleString();
  valDistance.textContent = (data.total_distance_m / 1000.0).toFixed(2) + " km";
  valConfidence.textContent = (data.mean_confidence * 100.0).toFixed(1) + "%";
  valLatency.textContent = data.execution_time_sec + "s";

  if (data.ground_truth_accuracy !== null && data.ground_truth_accuracy !== undefined) {
    valAccuracy.textContent = data.ground_truth_accuracy + "%";
    accuracyCard.classList.remove("hidden");
  } else {
    accuracyCard.classList.add("hidden");
  }

  // Draw Snapped Route Polyline
  const routeCoords = data.matched_coordinates.filter((c) => !isNaN(c[0]) && !isNaN(c[1]));
  if (routeCoords.length > 0) {
    const polyline = L.polyline(routeCoords, {
      color: "#10b981",
      weight: 5,
      opacity: 0.9,
      lineCap: "round",
      lineJoin: "round",
    }).addTo(clientState.snappedRouteLayer);

    const accHtml = data.ground_truth_accuracy ? `<br><b>Ground-Truth Acc:</b> ${data.ground_truth_accuracy}%` : "";
    polyline.bindPopup(`
      <div style="font-family: sans-serif; font-size: 0.8rem; line-height: 1.4;">
        <b style="color: #047857;">HMM Viterbi Matched Route</b><br>
        <b>Trip ID:</b> ${data.trip_id}<br>
        <b>Points:</b> ${data.point_count}<br>
        <b>Distance:</b> ${(data.total_distance_m / 1000.0).toFixed(2)} km<br>
        <b>Execution Time:</b> ${data.execution_time_sec}s
        ${accHtml}
      </div>
    `);

    clientState.map.fitBounds(polyline.getBounds(), { padding: [40, 40] });
  }

  // Traversed Arcs Table
  arcsTableBody.innerHTML = "";
  const uniqueArcs = data.unique_route_arcs || [];
  arcCountBadge.textContent = `${uniqueArcs.length} Arcs`;

  uniqueArcs.slice(0, 100).forEach((arcId, idx) => {
    const conf = data.confidences && data.confidences[idx] ? (data.confidences[idx] * 100).toFixed(1) + "%" : "-";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td style="font-family: var(--font-mono); font-weight: 600;">Arc #${arcId}</td>
      <td style="color: #059669; font-weight: 500;">${conf}</td>
    `;
    arcsTableBody.appendChild(tr);
  });
}

// 9. File Upload Handlers
function setupUploadHandlers() {
  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", handleFileSelection);

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelection();
    }
  });

  btnUploadMatch.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const tripId = uploadTripId.value.trim() || "00000007";
    showLoading(`Uploading ${file.name} to Host Server...`);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("trip_id", tripId);
      formData.append("partition", "student_test");

      const res = await fetch("/api/upload-and-match", {
        method: "POST",
        body: formData,
      });

      const result = await res.json();
      if (!res.ok) throw new Error(result.detail || "Upload matching failed");

      renderRawGps(result.raw_coordinates.map((c) => ({ lat: c[0], lon: c[1] })));
      renderMatchResults(result);
      logMessage(`File ${file.name} matched: ${result.point_count} pts, ${result.total_distance_m}m.`, "success");
    } catch (err) {
      showAlert(`Upload error: ${err.message}`);
      logMessage(`Upload failed: ${err.message}`, "error");
    } finally {
      hideLoading();
    }
  });
}

function handleFileSelection() {
  const file = fileInput.files[0];
  if (!file) return;
  dropZone.querySelector(".dropzone-title").innerHTML = `Selected: <b>${file.name}</b> (${(file.size / 1024).toFixed(1)} KB)`;
  btnUploadMatch.disabled = false;
  btnUploadMatch.focus();
}

// 10. Manual Coordinates Handler
function setupManualHandler() {
  btnManualMatch.addEventListener("click", async () => {
    const rawText = manualCoords.value.trim();
    if (!rawText) {
      showAlert("Please paste GPS coordinates into the box.");
      return;
    }

    try {
      showLoading("Parsing coordinates and sending to Host Server...");
      const lines = rawText.split("\n").map((l) => l.trim()).filter((l) => l);
      const coords = lines.map((line, idx) => {
        const parts = line.split(/[,\t]+/).map(Number);
        if (parts.length >= 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
          const lat = Math.abs(parts[0]) <= 90 ? parts[0] : parts[1];
          const lon = Math.abs(parts[0]) <= 90 ? parts[1] : parts[0];
          return { latitude: lat, longitude: lon, timestamp: parts[2] || idx };
        }
        throw new Error(`Invalid line: "${line}"`);
      });

      if (coords.length < 2) {
        throw new Error("Provide at least 2 coordinate rows.");
      }

      const tripId = manualTripId.value.trim() || "00000007";
      const payload = {
        trip_id: tripId,
        partition: "student_test",
        coordinates: coords,
      };

      const res = await fetch("/api/match", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const result = await res.json();
      if (!res.ok) throw new Error(result.detail || "Manual matching failed");

      renderRawGps(coords);
      renderMatchResults(result);
    } catch (err) {
      showAlert("Manual matching error: " + err.message);
    } finally {
      hideLoading();
    }
  });
}

// 11. Event Listeners
function setupEventListeners() {
  btnLoadSample.addEventListener("click", async () => {
    const selected = sampleTripSelect.value;
    if (!selected) return;
    const [partition, tripId] = selected.split("/");
    await loadTripData(partition, tripId);
  });

  btnMatchSample.addEventListener("click", async () => {
    if (!clientState.currentTripData) return;
    await executeMapMatch(clientState.currentTripData);
  });

  toggleRawGps.addEventListener("change", (e) => {
    if (e.target.checked) clientState.map.addLayer(clientState.rawGpsLayer);
    else clientState.map.removeLayer(clientState.rawGpsLayer);
  });

  toggleSnappedRoute.addEventListener("change", (e) => {
    if (e.target.checked) clientState.map.addLayer(clientState.snappedRouteLayer);
    else clientState.map.removeLayer(clientState.snappedRouteLayer);
  });

  if (btnClearLogs) {
    btnClearLogs.addEventListener("click", () => {
      if (logTerminal) logTerminal.innerHTML = "";
    });
  }

  if (alertClose) {
    alertClose.addEventListener("click", hideAlert);
  }
}

// UI Helpers
function showLoading(text) {
  loadingText.textContent = text || "Processing...";
  loadingOverlay.classList.remove("hidden");
}

function hideLoading() {
  loadingOverlay.classList.add("hidden");
}

function showAlert(msg) {
  alertMsg.textContent = msg;
  alertBox.classList.remove("hidden");
}

function hideAlert() {
  alertBox.classList.add("hidden");
}

// App Initialization
document.addEventListener("DOMContentLoaded", () => {
  initMap();
  setupTabs();
  setupUploadHandlers();
  setupManualHandler();
  setupEventListeners();
  loadSampleTrips();
  startHeartbeat();
  logMessage("📱 PathSnap Client Portal ready.", "system");
  logMessage("Connected to Host Server at 10.20.202.73:8000.", "info");

  window.addEventListener("resize", () => {
    if (clientState.map) clientState.map.invalidateSize();
  });
});
