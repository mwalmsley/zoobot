from zoobot import schemas

# use these by importing them in another script e.g.
# from zoobot.label_metadata import decals_label_cols 

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
decals_questions, decals_label_cols = schemas.extract_questions_and_label_cols(decals_pairs)

# subset of questions, useful for development/debugging
decals_partial_pairs = {
    'smooth-or-featured': ['_smooth', '_featured-or-disk'],
    'has-spiral-arms': ['_yes', '_no'],
    'bar': ['_strong', '_weak', '_no'],
    'bulge-size': ['_dominant', '_large', '_moderate', '_small', '_none']
}
decals_partial_questions, decals_partial_label_cols = schemas.extract_questions_and_label_cols(decals_partial_pairs)


gz2_pairs = {
    'smooth-or-featured': ['_smooth', '_featured-or-disk'],
    'disk-edge-on': ['_yes', '_no'],
    'has-spiral-arms': ['_yes', '_no'],
    'bar': ['_yes', '_no'],
    'bulge-size': ['_dominant', '_obvious', '_just-noticeable', '_no'],
    'something-odd': ['_yes', '_no'],
    'how-rounded': ['_round', '_in-between', '_cigar'],
    'bulge-shape': ['_round', '_boxy', '_no-bulge'],
    'spiral-winding': ['_tight', '_medium', '_loose'],
    'spiral-count': ['_1', '_2', '_3', '_4', '_more-than-4', '_cant-tell']
}
gz2_questions, gz2_label_cols = schemas.extract_questions_and_label_cols(gz2_pairs)

# useful for development/debugging
gz2_partial_pairs = {
    'smooth-or-featured': ['_smooth', '_featured-or-disk']
}
gz2_partial_questions, gz2_partial_label_cols = schemas.extract_questions_and_label_cols(gz2_partial_pairs)


def get_gz2_and_decals_dependencies(question_answer_pairs):
    """
    Get dict mapping each question (e.g. disk-edge-on) to the answer on which it depends (e.g. smooth-or-featured_featured-or-disk)

    Args:
        question_answer_pairs (dict): dict mapping questions (e.g. disk-edge-on) to list of answers (e.g. [disk-edge-on_yes, disk-edge-on_no])

    Returns:
        dict: see above
    """
    # will need to make your own func if the dependancies change

    #  awkward to keep the partial version as it changes these dependencies
    if 'disk-edge-on' in question_answer_pairs.keys():
        featured_branch_answer = 'disk-edge-on_no'
    else:
        featured_branch_answer = 'smooth-or-featured_featured-or-disk'  # skip it

    # luckily, these are the same in GZ2 and decals, just only some questions are asked
    dependencies = {
        'smooth-or-featured': None,  # always asked
        'disk-edge-on': 'smooth-or-featured_featured-or-disk',
        'has-spiral-arms': featured_branch_answer,
        'bar': featured_branch_answer,
        'bulge-size': featured_branch_answer,
        'something-odd': None,  # always asked
        'how-rounded': 'smooth-or-featured_smooth',
        'bulge-shape': 'disk-edge-on_yes',
        'edge-on-bulge': 'disk-edge-on_yes',
        'spiral-winding': 'has-spiral-arms_yes',
        'spiral-count': 'has-spiral-arms_yes',
        'spiral-arm-count': 'has-spiral-arms_yes', # bad naming...
        'merging': None
    }
    return dependencies


# decals_questions = [
#     'smooth-or-featured',
#     'disk-edge-on',
#     'has-spiral-arms',
#     'bar',
#     'bulge-size',
#     'how-rounded',
#     'edge-on-bulge',
#     'spiral-winding',
#     'spiral-arm-count',  # bad naming
#     'merging'
# ]

# decals_label_cols = [
#     'smooth-or-featured_smooth',
#     'smooth-or-featured_featured-or-disk',
#     'smooth-or-featured_artifact',
#     'disk-edge-on_yes',
#     'disk-edge-on_no',
#     'has-spiral-arms_yes',
#     'has-spiral-arms_no',
#     'bar_strong',
#     'bar_weak',
#     'bar_no',
#     #  5 answers for bulge size
#     'bulge-size_dominant',
#     'bulge-size_large',
#     'bulge-size_moderate',
#     'bulge-size_small',
#     'bulge-size_none',
#     'how-rounded_round',
#     'how-rounded_in-between',
#     'how-rounded_cigar-shaped',
#     'edge-on-bulge_boxy',
#     'edge-on-bulge_none',
#     'edge-on-bulge_rounded',
#     'spiral-winding_tight',
#     'spiral-winding_medium',
#     'spiral-winding_loose',
#     #  6 answers for spiral count
#     'spiral-arm-count_1',
#     'spiral-arm-count_2',
#     'spiral-arm-count_3',
#     'spiral-arm-count_4',
#     'spiral-arm-count_more-than-4',
#     'spiral-arm-count_cant-tell',
#     'merging_none',
#     'merging_minor-disturbance',
#     'merging_major-disturbance',
#     'merging_merger'
# ]


# decals_partial_questions = [
#     'smooth-or-featured',
#     'has-spiral-arms',
#     'bar',
#     'bulge-size'
# ]

# decals_partial_label_cols = [
#     'smooth-or-featured_smooth',
#     'smooth-or-featured_featured-or-disk',
#     'has-spiral-arms_yes',
#     'has-spiral-arms_no',
#     'spiral-winding_tight',
#     'spiral-winding_medium',
#     'spiral-winding_loose',
#     'bar_strong',
#     'bar_weak',
#     'bar_no',
#     'bulge-size_dominant',
#     'bulge-size_large',
#     'bulge-size_moderate',
#     'bulge-size_small',
#     'bulge-size_none'
# ]


# gz2_questions = [
#     'smooth-or-featured',
#     'disk-edge-on',
#     'has-spiral-arms',
#     'bar',
#     'bulge-size',
#     'something-odd',
#     'how-rounded',
#     'bulge-shape',
#     'spiral-winding',
#     'spiral-count'
# ]

# gz2_label_cols = [
#     'smooth-or-featured_smooth',
#     'smooth-or-featured_featured-or-disk',
#     'disk-edge-on_yes',
#     'disk-edge-on_no',
#     'has-spiral-arms_yes',
#     'has-spiral-arms_no',
#     'bar_yes',
#     'bar_no',
#     'bulge-size_dominant',
#     'bulge-size_obvious',
#     'bulge-size_just-noticeable',
#     'bulge-size_no',
#     'something-odd_yes',
#     'something-odd_no',
#     'how-rounded_round',
#     'how-rounded_in-between',
#     'how-rounded_cigar',
#     'bulge-shape_round',
#     'bulge-shape_boxy',
#     'bulge-shape_no-bulge',
#     'spiral-winding_tight',
#     'spiral-winding_medium',
#     'spiral-winding_loose',
#     'spiral-count_1',
#     'spiral-count_2',
#     'spiral-count_3',
#     'spiral-count_4',
#     'spiral-count_more-than-4',
#     'spiral-count_cant-tell'
# ]


# gz2_partial_questions = [
#     'smooth-or-featured'
#     # 'has-spiral-arms'
#     # 'bar',
#     # 'bulge-size'
# ]


# gz2_partial_label_cols = [
#     'smooth-or-featured_smooth',
#     'smooth-or-featured_featured-or-disk'
#     # 'has-spiral-arms_yes',
#     # 'has-spiral-arms_no'
#     # 'bar_yes',
#     # 'bar_no',
#     # 'bulge-size_dominant',
#     # 'bulge-size_obvious',
#     # 'bulge-size_just-noticeable',
#     # 'bulge-size_no'
# ]

