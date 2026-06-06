import requests

lat = 30.0668
lon = 77.7021

r = requests.get(
    "https://api.opentopodata.org/v1/srtm30m",
    params={
        "locations": f"{lat},{lon}"
    }
)

print(r.json())
