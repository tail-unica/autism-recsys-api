import os
import polars as pl

class Reader:
    def __init__(self, filepath: str):
        self._base = "turin_autism"
        self.inter = os.path.join(filepath, self._base + ".inter")
        self.item = os.path.join(filepath, self._base + ".item")
        self.user = os.path.join(filepath, self._base + ".user")
        self.item_link = os.path.join(filepath, self._base + ".item_link")
        self.user_link = os.path.join(filepath, self._base + ".user_link")
        
        self.kg = os.path.join(filepath, self._base + ".kg")

    def read_kg(self) -> pl.DataFrame:
        return pl.read_csv(self.kg, separator="\t", has_header=True)

    def link_item(self) -> pl.DataFrame:
        link = pl.read_csv(self.item_link, separator="\t", has_header=True)
        item = pl.read_csv(self.item, separator="\t", has_header=True)
        return link.join(item, on="poi_id:token", how="inner").drop("poi_id:token")

    def link_user(self) -> pl.DataFrame:
        link = pl.read_csv(self.user_link, separator="\t", has_header=True)
        user = pl.read_csv(self.inter, separator="\t", has_header=True)
        place = pl.read_csv(self.item_link, separator="\t", has_header=True)
        df = link.join(user, on="user_id:token", how="inner").drop("user_id:token")
        df = df.join(place, on="poi_id:token", how="inner").drop("poi_id:token")
        df = df.rename({"entity_id:token_right": "poi_id:token"})
        return df


    