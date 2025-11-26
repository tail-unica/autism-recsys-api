import polars as pl
import numpy as np
from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)

class Graph:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password)) 
        self.relations_map = {
            "sensoryFeature": "HAS_SENSORY_FEATURE", # place to sensory feature
            "category": "BELONGS_TO_CATEGORY", # place to category
            "sensoryCompatibility": "HAS_SENSORY_COMPATIBILITY", # user to sensory feature
            "preference": "HAS_PREFERENCE", # user to place category
            "placeCompatibility": "HAS_PLACE_COMPATIBILITY", # user to place
        }

    def close(self) -> None:
        self._driver.close()

    def create_categories(self) -> None:
        categories = {
            "Parchi": "Parchi",
            "Cultura_e_Musei": "Cultura e Musei",
            "Relax_e_Svago": "Relax e Svago",
            "Negozi_di_Fumetti": "Negozi di Fumetti",
            "Negozi_di_Tecnologia": "Negozi di Tecnologia",
            "Negozi_di_Abbigliamento": "Negozi di Abbigliamento",
            "Centri_Commerciali_e_Supermercati": "Centri Commerciali e Supermercati",
            "Biblioteche": "Biblioteche",
            "Librerie": "Librerie",
            "Sport": "Sport",
            "Bar_e_Pub": "Bar e Pub",
            "Ristoranti": "Ristoranti",
            "Gelaterie": "Gelaterie",
            "Piazze": "Piazze",
            "Stazioni": "Stazioni"
        }
        try:
            with self._driver.session() as session:

                cypher_query = """
                CREATE CONSTRAINT category_id IF NOT EXISTS
                FOR (c:Category)
                REQUIRE c.id IS UNIQUE
                """
                session.run(cypher_query)

                cats = [{"id": cid, "name": name} for cid, name in categories.items()]
                session.run(
                    """
                    UNWIND $categories AS category
                    CREATE (c:Category {id: category.id, name: category.name})
                    """,
                    categories=cats,
                )
        finally:
            logging.info("Categories created.")


    def create_sensory_features(self) -> None:
        sensory_features = ["LIGHT", "SPACE", "CROWD", "NOISE", "ODOR"]
        values = np.arange(1.0, 5.1, 0.1).tolist()

        try:
            with self._driver.session() as session:

                cypher_query = """
                CREATE CONSTRAINT sensory_feature_id IF NOT EXISTS
                FOR (sf:SensoryFeature)
                REQUIRE sf.id IS UNIQUE
                """
                session.run(cypher_query)

                for feature in sensory_features:
                    nodes = [{"id": f"{feature}.{value:.1f}", "feature": feature, "value": float(f"{value:.1f}")} for value in values]
                    session.run(
                    """
                    UNWIND $nodes AS node
                    CREATE (sf:SensoryFeature {id: node.id, feature: node.feature, value: node.value})
                    """,
                    nodes=nodes,
                    )
        finally:
            logging.info("Sensory features created.")

    def connect_sensory_feature(self, df: pl.DataFrame) -> None:
        sensory_features_relations = ["sensoryFeature", "sensoryCompatibility"]

        try:
            with self._driver.session() as session:

                # Create the first constraint for User
                constraint_user_query = """
                CREATE CONSTRAINT user_id IF NOT EXISTS
                FOR (u:User)
                REQUIRE u.id IS UNIQUE
                """
                # session.run(constraint_user_query)

                # Create the second constraint for Place
                constraint_place_query = """
                CREATE CONSTRAINT place_id IF NOT EXISTS
                FOR (p:Place)
                REQUIRE p.id IS UNIQUE
                """
                # session.run(constraint_place_query)

                filtered_df = df.filter(
                    pl.col("relation_id:token").is_in(sensory_features_relations)
                )
                # change relation ids using self.relations_map
                filtered_df = filtered_df.with_columns(
                    pl.col("relation_id:token")
                    .map_elements(lambda x: self.relations_map.get(x, x), return_dtype=pl.String)
                    .alias("relation_id:token")
                )
                # add columns head_type and tail_id
                filtered_df = filtered_df.with_columns(
                    pl.col("head_id:token").str.split(".").list.get(0).alias("_prefix"),
                    pl.col("tail_id:token").str.replace("sensory_feature.", "").alias("_feature_id"),
                ).with_columns(
                    pl.when(pl.col("_prefix") == "place")
                    .then(pl.lit("Place"))
                    .when(pl.col("_prefix") == "user")
                    .then(pl.lit("User"))
                    .otherwise(pl.lit("Unknown"))
                    .alias("head_type"),
                    pl.lit("SensoryFeature").alias("tail_type"),
                ).drop("_prefix")
                filtered_df = filtered_df.rename({
                    "head_id:token": "head_id_token",
                    "tail_id:token": "tail_id_token",
                    "relation_id:token": "relation_id_token",
                })
                # replace ODOUR with ODOR in _feature_type column
                filtered_df = filtered_df.with_columns(
                    pl.col("_feature_id").str.replace("ODOUR", "ODOR")
                )

                query = """
                UNWIND $rows AS row
                CALL apoc.merge.node([row.head_type], {id: row.head_id_token}) YIELD node AS head_node
                CALL apoc.merge.node([row.tail_type], {id: row._feature_id}) YIELD node AS tail_node
                CALL apoc.merge.relationship(head_node, row.relation_id_token, {}, {}, tail_node) YIELD rel AS r
                RETURN count(r) AS relationships_created
                """

                session.run(
                    query,
                    rows=filtered_df.to_dicts()
                )

        finally:
            logging.info("Sensory features connected.")

    def connect_categories(self, df: pl.DataFrame) -> None:
        category_relations = ["category", "preference"]

        try:
            with self._driver.session() as session:

                filtered_df = df.filter(
                    pl.col("relation_id:token").is_in(category_relations)
                )
                # change relation ids using self.relations_map
                filtered_df = filtered_df.with_columns(
                    pl.col("relation_id:token")
                    .map_elements(lambda x: self.relations_map.get(x, x), return_dtype=pl.String)
                    .alias("relation_id:token")
                )
                # add columns head_type and tail_type
                filtered_df = filtered_df.with_columns(
                    pl.col("head_id:token").str.split(".").list.get(0).alias("_prefix"),
                    pl.col("tail_id:token").str.split(".").list.get(1).alias("id_category"),
                ).with_columns(
                    pl.when(pl.col("_prefix") == "place")
                    .then(pl.lit("Place"))
                    .when(pl.col("_prefix") == "user")
                    .then(pl.lit("User"))
                    .otherwise(pl.lit("Unknown"))
                    .alias("head_type"),
                    pl.lit("Category").alias("tail_type"),
                ).drop("_prefix")
                filtered_df = filtered_df.rename({
                    "head_id:token": "head_id_token",
                    "tail_id:token": "tail_id_token",
                    "relation_id:token": "relation_id_token",
                })

                query = """
                UNWIND $rows AS row
                CALL apoc.merge.node([row.head_type], {id: row.head_id_token}) YIELD node AS head_node
                CALL apoc.merge.node([row.tail_type], {id: row.id_category}) YIELD node AS tail_node
                CALL apoc.merge.relationship(head_node, row.relation_id_token, {}, {}, tail_node) YIELD rel AS r
                RETURN count(r) AS relationships_created
                """

                session.run(
                    query,
                    rows=filtered_df.to_dicts()
                )

        except Exception as e:
            logging.error(f"An error occurred: {e}")

    def connect_place_compatibility(self, df: pl.DataFrame) -> None:
        place_compatibility_relations = ["placeCompatibility"]

        try:
            with self._driver.session() as session:

                filtered_df = df.filter(
                    pl.col("relation_id:token").is_in(place_compatibility_relations)
                )
                # change relation ids using self.relations_map
                filtered_df = filtered_df.with_columns(
                    pl.col("relation_id:token")
                    .map_elements(lambda x: self.relations_map.get(x, x), return_dtype=pl.String)
                    .alias("relation_id:token")
                )
                # add columns head_type and tail_type
                filtered_df = filtered_df.with_columns(
                    pl.lit("User").alias("head_type"),
                    pl.lit("Place").alias("tail_type"),
                )
                filtered_df = filtered_df.rename({
                    "head_id:token": "head_id_token",
                    "tail_id:token": "tail_id_token",
                    "relation_id:token": "relation_id_token",
                })

                query = """
                UNWIND $rows AS row
                CALL apoc.merge.node([row.head_type], {id: row.head_id_token}) YIELD node AS head_node
                CALL apoc.merge.node([row.tail_type], {id: row.tail_id_token}) YIELD node AS tail_node
                CALL apoc.merge.relationship(head_node, row.relation_id_token, {}, {}, tail_node) YIELD rel AS r
                RETURN count(r) AS relationships_created
                """

                session.run(
                    query,
                    rows=filtered_df.to_dicts()
                )

        finally:
            logging.info("Place compatibilities connected.")

    def place_metadata(self, df: pl.DataFrame) -> None:
        # define neo4j types for each column
        proprieties = {
            "entity_id": "STRING",
            "name": "STRING",
            "address": "STRING",
            "description": "STRING",
            "tags": "LIST<STRING>",
            "owner": "STRING",
            "updater": "STRING",
            "timestamp_insert": "DATETIME",
            "last_update": "DATETIME",
            "last_activity": "DATETIME",
            "coordinates": "POINT",
        }

        try:
            with self._driver.session() as session:
                # Extract column names before ":" and create rename mapping
                rename_mapping = {col: col.split(":")[0] for col in df.columns if ":" in col}
                df_filtered = df.rename(rename_mapping).select(list(proprieties.keys()))
                
                # Convert timestamps to datetime and parse coordinates
                df_filtered = df_filtered.with_columns([
                    pl.from_epoch(pl.col("timestamp_insert"), time_unit="s").alias("timestamp_insert"),
                    pl.from_epoch(pl.col("last_update"), time_unit="s").alias("last_update"),
                    pl.from_epoch(pl.col("last_activity"), time_unit="s").alias("last_activity"),
                ])
                
                # Convert to dicts and parse coordinates
                rows = df_filtered.to_dicts()
                for row in rows:
                    if row.get("coordinates"):
                        coords = row["coordinates"].split(", ")
                        row["coordinates"] = {"longitude": float(coords[0]), "latitude": float(coords[1])}

                query = """
                UNWIND $rows AS row
                CALL apoc.merge.node(['Place'], {id: row.entity_id}) YIELD node AS place_node
                SET place_node.name = row.name,
                    place_node.address = row.address,
                    place_node.description = row.description,
                    place_node.tags = row.tags,
                    place_node.owner = row.owner,
                    place_node.updater = row.updater,
                    place_node.timestamp_insert = datetime(row.timestamp_insert),
                    place_node.last_update = datetime(row.last_update),
                    place_node.last_activity = datetime(row.last_activity),
                    place_node.coordinates = point({longitude: row.coordinates.longitude, latitude: row.coordinates.latitude})
                RETURN count(place_node) AS places_updated
                """
                
                result = session.run(query, rows=rows)
                logging.info(f"Updated {result.single()['places_updated']} places.")
                
        finally:
            logging.info("Place metadata update completed.")


    def link_review(self, df: pl.DataFrame) -> None:
        # User IS_AUTHOR_OF Review ABOUT Place
        # Place HAS_REVIEW Review
        # DF columns: entity_id:token, poi_id:token, rating:float

        try:
            with self._driver.session() as session:

                filtered_df = df.select([
                    pl.col("entity_id:token").alias("entity_id"),
                    pl.col("poi_id:token").alias("poi_id"),
                    pl.col("rating:float").alias("rating"),
                ])

                query = """
                UNWIND $rows AS row
                CALL apoc.merge.node(['User'], {id: row.entity_id}) YIELD node AS user_node
                CALL apoc.merge.node(['Place'], {id: row.poi_id}) YIELD node AS place_node
                CREATE (review:Review {rating: row.rating})
                CREATE (user_node)-[:IS_AUTHOR_OF]->(review)-[:ABOUT]->(place_node)
                CREATE (place_node)-[:HAS_REVIEW]->(review)
                RETURN count(review) AS reviews_created
                """

                result = session.run(query,rows=filtered_df.to_dicts())
                logging.info(f"Created {result.single()['reviews_created']} reviews.")

        finally:
            logging.info("Reviews linked.")

    def user_metadata(self, df: pl.DataFrame) -> None:
        # define neo4j types for each column from .user file
        proprieties = {
            "entity_id": "STRING",
            "autism": "BOOLEAN",
        }
        print(df)
        try:
            with self._driver.session() as session:
                # Extract column names before ":" and create rename mapping
                rename_mapping = {col: col.split(":")[0] for col in df.columns if ":" in col}
                df_filtered = df.rename(rename_mapping).select(list(proprieties.keys()))
                
                # Convert to dicts
                rows = df_filtered.to_dicts()

                query = """
                UNWIND $rows AS row
                CALL apoc.merge.node(['User'], {id: row.entity_id}) YIELD node AS user_node
                SET user_node.autism = row.autism
                RETURN count(user_node) AS users_updated
                """
                
                result = session.run(query, rows=rows)
                logging.info(f"Updated {result.single()['users_updated']} users.")
                
        finally:
            logging.info("User metadata update completed.")