import neo4j

def _parse_geojson_from_neo4j_point(point: str, properties: dict = None) -> dict:
    """Parse a SRID string into a GeoJSON dictionary."""
    try:
        point = point.replace("point({", "")
        point = point.replace("})", "")
        data = point.split(", ")
        lon = float(data[1].split(": ")[1])
        lat = float(data[2].split(": ")[1])
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": properties or {},
        }
    except Exception:
        return None

def fetch_place_info(session: neo4j.Session, info: str) -> dict:
    """Fetch place information from the database based on the exact provided place name."""
    result = session.run(
        "MATCH (p:Place {name: $info}) "
        "OPTIONAL MATCH (p)-[:BELONGS_TO_CATEGORY]->(c:Category) "
        "OPTIONAL MATCH (p)-[:HAS_SENSORY_FEATURE]->(sf:SensoryFeature) "
        "RETURN p.name AS name, p.address AS address, p.coordinates AS coordinates, "
        "c.id AS category_id, "
        "collect({feature_name: sf.feature, rating: sf.value}) AS sensory_features",
        info=info,
    )
    record = result.single()
    if record:
        return {
            "place": record["name"],
            "category": record["category_id"],
            "address": record["address"],
            "coordinates": _parse_geojson_from_neo4j_point(
                record["coordinates"],
                properties={"name": record["name"]}
            ) if record["coordinates"] else None,
            "sensory_features": [
                {
                    "feature_name": sf["feature_name"].lower(),
                    "rating": sf["rating"]
                }
                for sf in record["sensory_features"]
                if sf["feature_name"] is not None
            ],
        }
    else:
        return {}