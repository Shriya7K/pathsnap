"""Integration and unit tests for PathSnap FastAPI Backend."""
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


class TestPathSnapAPI(unittest.TestCase):

    def test_01_health_endpoint(self):
        """Test GET /api/health returns healthy and model loaded."""
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertTrue(data["model_loaded"])
        self.assertGreater(data["available_student_test_trips"], 0)

    def test_02_list_trips_endpoint(self):
        """Test GET /api/trips returns sample trips."""
        response = client.get("/api/trips")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("trips", data)
        self.assertGreater(len(data["trips"]), 0)

        # Check trip structure
        first_trip = data["trips"][0]
        self.assertIn("trip_id", first_trip)
        self.assertIn("partition", first_trip)
        self.assertIn("point_count", first_trip)

    def test_03_get_sample_trip_trajectory(self):
        """Test GET /api/trip/{partition}/{trip_id} returns coordinates."""
        response = client.get("/api/trip/student_test/00000001")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["trip_id"], "00000001")
        self.assertGreater(len(data["coordinates"]), 0)
        coord = data["coordinates"][0]
        self.assertIn("latitude", coord)
        self.assertIn("longitude", coord)

    def test_04_map_match_endpoint_success(self):
        """Test POST /api/match returns matched route with real model & HMM."""
        # Fetch 20 points from student test trip 00000001
        trip_resp = client.get("/api/trip/student_test/00000001")
        coords = trip_resp.json()["coordinates"][:30]

        payload = {
            "trip_id": "00000001",
            "partition": "student_test",
            "coordinates": coords,
        }

        response = client.post("/api/match", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["trip_id"], "00000001")
        self.assertEqual(data["point_count"], len(coords))
        self.assertEqual(len(data["matched_arcs"]), len(coords))
        self.assertEqual(len(data["matched_coordinates"]), len(coords))
        self.assertGreater(data["total_distance_m"], 0.0)
        self.assertGreater(data["mean_confidence"], 0.0)
        self.assertGreater(len(data["unique_route_arcs"]), 0)

    def test_05_coordinate_validation_error(self):
        """Test POST /api/match rejects invalid latitude > 90."""
        payload = {
            "trip_id": "00000001",
            "partition": "student_test",
            "coordinates": [
                {"latitude": 150.0, "longitude": 10.0},
                {"latitude": 51.5, "longitude": 10.1},
            ],
        }
        response = client.post("/api/match", json=payload)
        self.assertEqual(response.status_code, 422)
        err = response.json()
        self.assertIn("detail", err)

    def test_06_insufficient_points_error(self):
        """Test POST /api/match rejects trajectories with fewer than 2 points."""
        payload = {
            "trip_id": "00000001",
            "partition": "student_test",
            "coordinates": [
                {"latitude": 51.5, "longitude": 10.0},
            ],
        }
        response = client.post("/api/match", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_07_file_upload_matching(self):
        """Test POST /api/upload-and-match with simulated TSV file."""
        trip_resp = client.get("/api/trip/train/00000001")
        coords = trip_resp.json()["coordinates"][:15]
        tsv_lines = [f"{c['longitude']}\t{c['latitude']}\t{c['timestamp']}" for c in coords]
        tsv_content = "\n".join(tsv_lines) + "\n"

        files = {"file": ("test_00000001.track", tsv_content.encode("utf-8"), "text/plain")}
        data = {"trip_id": "00000001", "partition": "train"}

        response = client.post("/api/upload-and-match", files=files, data=data)
        if response.status_code != 200:
            print("Upload error detail:", response.json())
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertEqual(res_data["point_count"], len(coords))
        self.assertEqual(len(res_data["matched_arcs"]), len(coords))


if __name__ == "__main__":
    unittest.main()
