"""
Utilities for data loading in experiments.
Load the data from Neo4j database into appropriate structures for hopwise.
"""
import os

import neo4j
import polars as pl

def parse_id(token: str) -> str:
    return token.replace(":", "_").replace("-", "_")

def load_data(session: neo4j.Session, path: str) -> None:
    """Load data from Neo4j and save as CSV files with tab separator."""
    os.makedirs(path, exist_ok=True)
    
    parse_df(get_kg(session, yield_custom_id=["SensoryFeature", "Category"])).write_csv(os.path.join(path, "autism.kg"), separator="\t")
    parse_df(get_inter(session)).write_csv(os.path.join(path, "autism.inter"), separator="\t")
    parse_df(get_user(session)).write_csv(os.path.join(path, "autism.user"), separator="\t")
    parse_df(get_item(session)).write_csv(os.path.join(path, "autism.item"), separator="\t")
    parse_df(get_user_link(session)).write_csv(os.path.join(path, "autism.user_link"), separator="\t")
    parse_df(get_item_link(session)).write_csv(os.path.join(path, "autism.item_link"), separator="\t")

def parse_df(df: pl.DataFrame) -> pl.DataFrame:
    """Parse DataFrame columns before creating CSV files.
    Drop columns with complex structures, convert float and int to appropriate types,
    convert everything else to string."""
    
    for col in df.columns:
        dtype = df[col].dtype
        
        # Skip if already correct type
        if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Utf8]:
            continue
        
        # Convert lists to comma-separated strings
        if dtype == pl.List:
            df = df.with_columns(
                pl.col(col).list.eval(pl.element().cast(pl.Utf8)).list.join(", ").alias(col)
            )
            continue
        
        # Drop complex types (structs, objects)
        if dtype in [pl.Struct, pl.Object]:
            df = df.drop(col)
            continue
        
        # Try to convert to string for other types
        try:
            df = df.with_columns(pl.col(col).cast(pl.Utf8))
        except:
            # If conversion fails, drop the column
            df = df.drop(col)
    
    return df

def get_kg(session: neo4j.Session, yield_custom_id: list[str] = None) -> pl.DataFrame:
    # return every relation in the neo4j database as a triple (source, relation, target)
    import hashlib
    
    allowed_relations = [
        "HAS_SENSORY_FEATURE",
        "HAS_SENSORY_COMPATIBILITY",
        "BELONGS_TO_CATEGORY",
        "HAS_PLACE_COMPATIBILITY",
        "HAS_PREFERENCE"
    ]
    query = """
        MATCH (a)-[r]->(b)
        WHERE type(r) IN $allowed_relations
        RETURN DISTINCT 
            elementId(a) AS source_id, 
            labels(a)[0] AS source_type, 
            type(r) AS relation, 
            elementId(b) AS target_id, 
            labels(b)[0] AS target_type
    """
    result = session.run(query, allowed_relations=allowed_relations)

    id_map = {}
    if yield_custom_id is not None:
        labels_filter = "|".join(yield_custom_id)
        query_id = f"""
            MATCH (n:{labels_filter})
            RETURN elementId(n) AS node_id, n.id AS custom_id
        """
        result_id = session.run(query_id)
        id_map = {record["node_id"]: record["custom_id"] for record in result_id}
    
    data = []
    for record in result:
        def parse_id_if_custom(node_id):
            if yield_custom_id is not None and node_id in id_map:
                return str(id_map[node_id])
            else:
                return str(parse_id(node_id))
        source_token = f"{record['source_type']}.{parse_id_if_custom(record['source_id'])}"
        target_token = f"{record['target_type']}.{parse_id_if_custom(record['target_id'])}"
        data.append({
            "head_id:token": source_token,
            "relation_id:token": record["relation"],
            "tail_id:token": target_token
        })
    
    return pl.DataFrame(data)

def get_inter(session: neo4j.Session) -> pl.DataFrame:
    query = """
        MATCH (u:User)-[:IS_AUTHOR_OF]->(r:Review)-[:ABOUT]->(p:Place)
        RETURN DISTINCT 
            elementId(u) AS user_id, 
            elementId(p) AS place_id,
            r.rating AS rating
    """
    result = session.run(query)
    
    data = []
    for record in result:
        user_token = f"{parse_id(record['user_id'])}"
        place_token = f"{parse_id(record['place_id'])}"
        data.append({
            "user_id:token": user_token,
            "poi_id:token": place_token,
            "rating:float": float(record['rating'])
        })
    
    return pl.DataFrame(data)

def get_attr(session: neo4j.Session, entity_type: str) -> pl.DataFrame:
    """
    Get attributes for users or items.
    
    Args:
        session: Neo4j session
        entity_type: Either 'user' or 'item'
    """
    if entity_type.lower() == 'user':
        label = 'User'
        token_field = 'user_id:token'
    elif entity_type.lower() == 'item':
        label = 'Place'
        token_field = 'poi_id:token'
    else:
        raise ValueError("entity_type must be either 'user' or 'item'")
    
    query = f"""
        MATCH (n:{label})
        RETURN DISTINCT 
            elementId(n) AS entity_id,
            properties(n) AS attributes
    """
    result = session.run(query)
    
    data = []
    column_types = {}  # Track the type suffix for each key
    
    for record in result:
        entity_token = f"{parse_id(record['entity_id'])}"
        row = {token_field: entity_token}
        
        # Add each attribute as a separate column
        if record['attributes']:
            for key, value in record['attributes'].items():
                if key == 'id':
                    continue
                
                # Determine the new type for this key
                if isinstance(value, float):
                    new_type = 'float'
                elif isinstance(value, bool):
                    new_type = 'token'
                elif isinstance(value, int):
                    new_type = 'int'
                elif isinstance(value, list):
                    new_type = 'token_seq'
                elif isinstance(value, str) and ' ' in value:
                    new_type = 'token_seq'
                else:
                    new_type = 'token'
                
                # Check if we've seen this key before and resolve the final column type
                if key in column_types:
                    existing_type = column_types[key]
                    final_type = existing_type
                    # token_seq has priority over everything
                    if new_type == 'token_seq' or existing_type == 'token_seq':
                        final_type = 'token_seq'
                    # token has priority over float and int
                    elif new_type == 'token' or existing_type == 'token':
                        final_type = 'token'
                    # float has priority over int
                    elif new_type == 'float' or existing_type == 'float':
                        final_type = 'float'
                    # If the resolved type changed, align previous rows with the new column name
                    if final_type != existing_type:
                        old_column = f"{key}:{existing_type}"
                        new_column = f"{key}:{final_type}"
                        for previous_row in data:
                            if old_column in previous_row:
                                previous_row[new_column] = previous_row.pop(old_column)
                    column_types[key] = final_type
                else:
                    column_types[key] = new_type
                
                final_type = column_types[key]
                row[f"{key}:{final_type}"] = value
        
        data.append(row)
    
    return pl.DataFrame(data)

def get_link(session: neo4j.Session, entity_type: str = 'user') -> pl.DataFrame:
    if entity_type.lower() == 'user':
        label = 'User'
        token_field = 'user_id:token'
    elif entity_type.lower() == 'item':
        label = 'Place'
        token_field = 'poi_id:token'
    else:
        raise ValueError("entity_type must be either 'user' or 'item'")
    
    query = f"""
        MATCH (n:{label})
        RETURN DISTINCT 
            elementId(n) AS entity_id
    """
    result = session.run(query)
    
    data = []
    for record in result:
        entity_token = f"{parse_id(record['entity_id'])}"
        data.append({
            token_field: entity_token,
            "entity_id:token": f"{label}.{entity_token}"
        })
    
    return pl.DataFrame(data)

def get_user(session: neo4j.Session) -> pl.DataFrame:
    return get_attr(session, 'user')

def get_item(session: neo4j.Session) -> pl.DataFrame:
    return get_attr(session, 'item')

def get_user_link(session: neo4j.Session) -> pl.DataFrame:
    return get_link(session, 'user')

def get_item_link(session: neo4j.Session) -> pl.DataFrame:
    return get_link(session, 'item')