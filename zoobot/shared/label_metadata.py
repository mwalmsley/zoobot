
# mostly refactored out into galaxy-datasets.shared.label_metadata.
# zoobot depends on galaxy-datasets, but galaxy-datasets does not depend on zoobot

from galaxy_datasets.shared.label_metadata import extract_questions_and_label_cols



"""
Non-orthogonal schemas (i.e. without the -dr5, -gz2, etc.)
Not recommended for use with ML due to potential overlap between questions from different surveys.
May still be convenient for exploring catalogs (just to save typing -dr5 etc all the time)
"""

decals_pairs = {
    'smooth-or-featured': ['_smooth', '_featured-or-disk', '_artifact'],
    'disk-edge-on': ['_yes', '_no'],
    'has-spiral-arms': ['_yes', '_no'],
    'bar': ['_strong', '_weak', '_no'],
    'bulge-size': ['_dominant', '_large', '_moderate', '_small', '_none'],
    'how-rounded': ['_round', '_in-between', '_cigar-shaped'],
    'edge-on-bulge': ['_boxy', '_none', '_rounded'],
    'spiral-winding': ['_tight', '_medium', '_loose'],
    'spiral-arm-count': ['_1', '_2', '_3', '_4', '_more-than-4', '_cant-tell'],
    'merging': ['_none', '_minor-disturbance', '_major-disturbance', '_merger']
}
decals_questions, decals_label_cols = extract_questions_and_label_cols(decals_pairs)


gz2_pairs = {
    'smooth-or-featured': ['_smooth', '_featured-or-disk', '_artifact'],
    'disk-edge-on': ['_yes', '_no'],
    'has-spiral-arms': ['_yes', '_no'],
    'bar': ['_yes', '_no'],
    'bulge-size': ['_dominant', '_obvious', '_just-noticeable', '_no'],
    'something-odd': ['_yes', '_no'],
    'how-rounded': ['_round', '_in-between', '_cigar'],
    'bulge-shape': ['_round', '_boxy', '_no-bulge'],
    'spiral-winding': ['_tight', '_medium', '_loose'],
    'spiral-arm-count': ['_1', '_2', '_3', '_4', '_more-than-4', '_cant-tell']
}
gz2_questions, gz2_label_cols = extract_questions_and_label_cols(gz2_pairs)


# useful for development/debugging
gz2_partial_ortho_pairs = {
    'smooth-or-featured-gz2': ['_smooth', '_featured-or-disk']
}
gz2_partial_ortho_questions, gz2_partial_ortho_label_cols = extract_questions_and_label_cols(gz2_partial_ortho_pairs)

