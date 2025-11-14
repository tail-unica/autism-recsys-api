import neo4j

def _parse_geojson_from_neo4j_point(point: neo4j.spatial.Point, properties: dict = None) -> dict:
    """Parse a Neo4j Point object into a GeoJSON dictionary."""
    try:
        lon, lat = point.x, point.y
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": properties or {},
        }
    except AttributeError:
        return None

def fetch_place_info(session: neo4j.Session, info: str, logger = None) -> dict:
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
    if logger:
        logger.info(f"Fetched place info for '{info}': {record}")
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