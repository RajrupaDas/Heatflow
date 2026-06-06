import requests

r = requests.get(
    "https://macrostrat.org/api/v2/defs/sources"
)

j = r.json()

print(type(j))
print(j.keys())

print(str(j)[:1000])
