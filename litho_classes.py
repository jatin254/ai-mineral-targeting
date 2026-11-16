"""Lithology class mapping for the GSI 50K map (KA/AP block). Edit here, not in the build scripts."""

CLASSES = [  # code, name, rock types (GSI 'lithologic' values, upper case)
    (1, "mafic_volcanic", ["META-BASALT", "PILLOWED METABASALT", "AMPHIBOLITE", "META-TUFF",
                           "META-ANDESITE", "ANDESITE"]),
    (2, "ultramafic", ["META-ULTRAMAFITE", "ULTRAMAFITE", "BASALTIC KOMATIITE", "PYROXENITE",
                       "META-PYROXENITE", "TALC TREMOLITE SCHIST", "KIMBERLITE"]),
    (3, "bif_chert", ["BANDED IRON FORMATION", "BANDED FERRUGINOUS CHERT", "BANDED MAGNETITE QUARTZITE",
                      "BANDED FERRUGINOUS QUARTZITE", "BANDED HEMATITE QUARTZITE", "MANGANIFEROUS CHERT",
                      "MANGANIFEROUS PHYLLITE"]),
    (4, "chl_ser_schist", ["CHLORITE SCHIST", "CHLORITE SERICITE SCHIST", "SERICITE SCHIST"]),
    (5, "metasediment", ["ARGILLITE", "GRAYWACKE", "PHYLLITE", "CARBONACEOUS PHYLLITE", "CONGLOMERATE",
                         "POLYMICTIC CONGLOMERATE", "QUARTZITE", "FUCHSITE QUARTZITE",
                         "SILLIMANITE-KYANITE-STAUROLITE-GARNET SCHIST", "MICA SCHIST / SCHIST",
                         "ANDALUSITE SCHIST", "QUARTZ-KYANITE SCHIST", "DOLOMITE", "FERRODOLOMITE",
                         "LIMESTONE", "DOLOMITIC LIMESTONE", "ARENITE", "FERRO DOLOMITE"]),
    (6, "felsic_volcanic", ["META-RHYOLITE", "META-ACID VOLCANICS", "QUARTZ PORPHYRY", "META-DACITE"]),
    (7, "gneiss", ["MIGMATITE GNEISS", "GREY HORNBLENDE BIOTITE GNEISS", "BIOTITE GNEISS",
                   "GREY HORNBLENDE GNEISS", "GREY BIOTITE GNEISS", "GNEISS", "GRANITE GNEISS", "PINK GNEISS",
                   "HORNBLENDE BIOTITE GRANITE GNEISS"]),
    (8, "granitoid", ["GREY GRANITE", "GREY HORNBLENDE BIOTITE GRANITE", "PINK PORPHYRITIC GRANITE",
                      "GREY BIOTITE GRANITE", "PINK HORNBLENDE BIOTITE GRANITE", "ALKALI FELDSPAR GRANITE",
                      "GRANITOID", "PINK GRANITE", "GRANITE", "PINK HORNBLENDE GRANITE", "PINK BIOTITE GRANITE",
                      "PORPHYRITIC GRANITE", "HORNBLENDE GRANITE", "LEUCO GRANITE", "GREY HORNBLENDE GRANITE",
                      "PEGMATITE", "SYENITE", "PINK SYENITE", "DIORITE", "APLITE"]),
    (9, "mafic_intrusive", ["DOLERITE", "GABBRO", "META-GABBRO", "OLIVINE GABBRO", "ANORTHOSITE GABBRO"]),
    (10, "quartz_vein", ["QUARTZ VEIN/REEF"]),
    (11, "cover", ["LATERITE", "GRAVEL (TERRACE)", "CALCRETE"]),
]
OTHER = (12, "other")  # anything not listed above

# Thin units: represented through distance layers in build_vector_features.py, not the class map.
THIN = {"ultramafic", "bif_chert", "chl_ser_schist", "mafic_intrusive", "quartz_vein", "felsic_volcanic"}

# Greenstone (supracrustal) belt membership, from supergroup / group attributes
GREENSTONE_SUPERGROUP_KEYS = ["DHARWAR"]
GREENSTONE_GROUP_KEYS = ["SARGUR", "OLDER SUPRACRUSTAL", "LAYERED ULTRAMAFIC"]

LOOKUP = {rock: (code, name) for code, name, rocks in CLASSES for rock in rocks}


def classify(lithologic) -> tuple[int, str]:
    key = str(lithologic or "").strip().upper()
    return LOOKUP.get(key, OTHER)


def is_greenstone(supergroup, group_name) -> bool:
    sg, gr = str(supergroup or "").upper(), str(group_name or "").upper()
    return any(k in sg for k in GREENSTONE_SUPERGROUP_KEYS) or any(k in gr for k in GREENSTONE_GROUP_KEYS)
