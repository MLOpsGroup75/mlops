curl -X POST "http://localhost:8000/v1/predict"   -H "Content-Type: application/json"   -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housingMedianAge": 41.0,
    "totalRooms": 880.0,
    "totalBedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "medianIncome": 8.3252,
    "medianHouseValue": 452600.0,
    "oceanProximity": "NEAR BAY"
  }'

