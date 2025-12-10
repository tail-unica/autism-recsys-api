import neo4j
from typing import List
from src.core.logger import logger

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

def _parse_lon_lat_from_geojson(geojson: dict) -> tuple:
    """Extract longitude and latitude from a GeoJSON dictionary."""
    try:
        coordinates = geojson.get("geometry", {}).get("coordinates", [])
        if len(coordinates) == 2:
            lon, lat = coordinates
            return lon, lat
    except (AttributeError, TypeError):
        pass
    return None, None

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
        logger.warning(f"No place found with name '{info}'")
        return None

def search(
    session: neo4j.Session,
    query: str,
    limit: int = 10,
    position: dict = None,
    distance: float = 1000.0, # in meters
    categories: list = None
) -> List[dict]:
    """Returns a list of places names based on the search query and optional filters.
    Integra una seconda fase di ricerca semantica (full-text) quando query non è vuoto e i risultati iniziali sono insufficienti.
    Richiede un indice full-text creato a parte:
    CALL db.index.fulltext.createNodeIndex('placeNameFulltext',['Place'],['name']);
    """
    normalized_query = query.strip() if query else ""
    cypher = "MATCH (p:Place) "
    params = {}
    where_clauses = []

    if normalized_query:
        where_clauses.append("toLower(p.name) CONTAINS toLower($query)")
        params["query"] = normalized_query

    if categories:
        categories = [cat.lower() for cat in categories]
        where_clauses.append("EXISTS { (p)-[:BELONGS_TO_CATEGORY]->(c:Category) WHERE toLower(c.id) IN $categories }")
        params["categories"] = categories

    if position:
        lon, lat = _parse_lon_lat_from_geojson(position)
        if lon is not None and lat is not None:
            where_clauses.append(
                "point.distance(p.coordinates, point({longitude: $lon, latitude: $lat})) <= $distance"
            )
            params["lon"], params["lat"] = lon, lat
            params["distance"] = distance

    if where_clauses:
        cypher += "WHERE " + " AND ".join(where_clauses) + " "

    # TODO: improve randomization strategy for large datasets
    if not normalized_query:
        logger.info("No search query provided, returning random places")
        cypher += "WITH p ORDER BY rand() "

    cypher += (
        "RETURN p.name AS name "
        "LIMIT $limit"
    )
    params["limit"] = limit
    logger.info(f"Querying (primary): {cypher} with params: {params}")

    # FIX: evitare conflitto 'query' con argomento posizionale di Session.run()
    result = session.run(cypher, parameters=params)
    names = [record["name"] for record in result]

    # Ricerca semantica (full-text) se query presente e risultati incompleti
    if normalized_query and len(names) < limit:
        remaining = limit - len(names)
        if remaining > 0:
            semantic_cypher = (
                "CALL db.index.fulltext.queryNodes('placeNameFulltext', $query) "
                "YIELD node, score "
                "RETURN node.name AS name "
                "LIMIT $remaining"
            )
            logger.info(f"Querying (semantic fallback): {semantic_cypher} with params: {{'query': '{normalized_query}', 'remaining': {remaining}}}")
            try:
                semantic_result = session.run(
                    semantic_cypher,
                    parameters={"query": normalized_query, "remaining": remaining}
                )
                seen = set(names)
                for record in semantic_result:
                    n = record["name"]
                    if n not in seen:
                        names.append(n)
                        seen.add(n)
            except neo4j.exceptions.ClientError as e:
                # No index found or other error
                logger.warning(f"Semantic search skipped (full-text index missing or error): {e}")

    # TODO: futura estensione: indice vettoriale per embedding semantiche (Neo4j vector indexes)
    return [{"name": n} for n in names]