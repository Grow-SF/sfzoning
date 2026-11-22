"""The fields of the lots and districts downloads: data code (it shapes the data files) that the
README table also shows. A change here re-runs ogr2ogr."""

# (name in the data, name in .shp files, meaning); shapefile names are limited to 10 characters.
LOT_FIELDS = [
    ("blklot", "blklot", "DataSF parcel id: block and lot"),
    ("code", "code", "zoning district code read from the map"),
    ("height_code", "height", "height district code (where the period has a height map)"),
    ("category", "category", "category used on sfzoning.com"),
    ("confidence", "confidence", "0 to 1, how sure the reading is"),
    ("source_sheet", "src_sheet", "sheet the label was read from"),
    ("note", "note", "why a park lot shows as public, if so"),
]
DISTRICT_FIELDS = [
    ("code", "code", "zoning district code read from the map"),
    ("category", "category", "category used on sfzoning.com"),
]
