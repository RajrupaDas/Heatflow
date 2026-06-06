import requests

lat = 30.0668
lon = 77.7021

url = (
    f"https://macrostrat.org/api/v2/geologic_units/map"
    f"?lat={lat}&lng={lon}"
)

r = requests.get(url)

print(r.status_code)
print(r.text[:1000])
