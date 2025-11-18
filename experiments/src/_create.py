from neo4j import GraphDatabase
import polars as pl

class Graph:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password)) 
        self.sensory_features = ["LIGHT", "SPACE", "CROWD", "NOISE", "ODOUR"]
        self.relations_map = {
            "sensoryFeature": "HAS_SENSORY_FEATURE", # place to sensory feature
            "category": "BELONGS_TO_CATEGORY", # place to category
            "sensoryCompatibility": "HAS_SENSORY_COMPATIBILITY", # user to sensory feature
            "preference": "HAS_PREFERENCE", # user to place category
            "placeCompatibility": "HAS_PLACE_COMPATIBILITY", # user to place
        }

    def create_sensory_features(self) -> None:
        try:
            with self._driver.session() as session:

                cypher_query = """
                CREATE CONSTRAINT sensory_feature_id IF NOT EXISTS
                FOR (sf:SensoryFeature)
                REQUIRE sf.id IS UNIQUE
                """
                session.run(cypher_query)

                features = [{"id": feature} for feature in self.sensory_features]
                session.run(
                    """
                    UNWIND $features AS feature
                    CREATE (sf:SensoryFeature {id: feature.id})
                    """,
                    features=features,
                )              

        finally:
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

                cats = [{"id": cat_id, "name": cat_name} for cat_id, cat_name in categories.items()]
                session.run(
                    """
                    UNWIND $categories AS category
                    CREATE (c:Category {id: category.id, name: category.name})
                    """,
                    categories=cats,
                )
        finally:
            self._driver.close()

    def populate_from_kg(self, df: pl.DataFrame) -> None:
        try:
            sensory_features_relations = ["sensoryFeature", "sensoryCompatibility"]

            # sensory features
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

                # sensory features
                filtered_df = df.filter(
                    pl.col("relation_id:token").is_in(sensory_features_relations)
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
                    pl.col("tail_id:token").str.split(".").list.get(1).alias("_feature_type"),
                    # Convert the two parts to a float value
                    (pl.col("tail_id:token").str.split(".").list.get(2).cast(pl.Float64) +
                    pl.col("tail_id:token").str.split(".").list.get(3).cast(pl.Float64) / 10.0)
                    .alias("_value"),
                ).with_columns(
                    pl.when(pl.col("_prefix") == "place")
                    .then(pl.lit("Place"))
                    .when(pl.col("_prefix") == "user")
                    .then(pl.lit("User"))
                    .otherwise(pl.lit("Unknown"))
                    .alias("head_type"),
                    pl.lit("SensoryFeature").alias("tail_type"),
                ).drop("_prefix")
                # rename columns
                filtered_df = filtered_df.rename({
                    "head_id:token": "head_id_token",
                    "tail_id:token": "tail_id_token",
                    "relation_id:token": "relation_id_token",
                })
                # replace ODOUR with ODOR in _feature_type column
                filtered_df = filtered_df.with_columns(
                    pl.col("_feature_type").str.replace("ODOUR", "ODOR")
                )

                print(filtered_df)

                _query = """
                UNWIND $rows AS row
                CALL apoc.merge.node([row.head_type], {id: row.head_id_token}) YIELD node AS head_node
                CALL apoc.merge.node([row.tail_type], {id: row._feature_type}) YIELD node AS tail_node
                CALL apoc.merge.relationship(head_node, row.relation_id_token, {value: row._value}, {}, tail_node) YIELD rel
                RETURN count(rel)
                """

                session.run(_query, rows=filtered_df.to_dicts())

            with self._driver.session() as session:
                filtered_df = df.filter(
                    pl.col("relation_id:token").is_not_in(sensory_features_relations)
                )

        finally:
            self.close()

    def close(self) -> None:
        """If a persistent driver was stored on the instance, close it."""
        d = getattr(self, "_driver", None)
        if d:
            try:
                d.close()
            except Exception:
                pass