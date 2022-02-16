from zoobot.shared import schemas

# use these by importing them in another script e.g.
# from zoobot.label_metadata import decals_label_cols 

# subset of questions, useful for development/debugging
decals_partial_pairs = {
    'smooth-or-featured': ['_smooth', '_featured-or-disk'],
    'has-spiral-arms': ['_yes', '_no'],
    'bar': ['_strong', '_weak', '_no'],
    'bulge-size': ['_dominant', '_large', '_moderate', '_small', '_none']
}
decals_partial_questions, decals_partial_label_cols = schemas.extract_questions_and_label_cols(decals_partial_pairs)

# dr5, implicitly
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

# The "ortho" versions avoid mixing votes from different campaigns
decals_dr5_ortho_pairs = {
    'smooth-or-featured-dr5': ['_smooth', '_featured-or-disk', '_artifact'],
    'disk-edge-on-dr5': ['_yes', '_no'],
    'has-spiral-arms-dr5': ['_yes', '_no'],
    'bar-dr5': ['_strong', '_weak', '_no'],
    'bulge-size-dr5': ['_dominant', '_large', '_moderate', '_small', '_none'],
    'how-rounded-dr5': ['_round', '_in-between', '_cigar-shaped'],
    'edge-on-bulge-dr5': ['_boxy', '_none', '_rounded'],
    'spiral-winding-dr5': ['_tight', '_medium', '_loose'],
    'spiral-arm-count-dr5': ['_1', '_2', '_3', '_4', '_more-than-4', '_cant-tell'],
    'merging-dr5': ['_none', '_minor-disturbance', '_major-disturbance', '_merger']
}
decals_dr5_ortho_questions, decals_dr5_ortho_label_cols = schemas.extract_questions_and_label_cols(decals_dr5_ortho_pairs)

# exactly the same for dr8. 
decals_dr8_ortho_pairs = decals_dr5_ortho_pairs.copy()
for question, answers in decals_dr8_ortho_pairs.copy().items(): # avoid modifying while looping
    decals_dr8_ortho_pairs[question.replace('-dr5', '-dr8')] = answers
    del decals_dr8_ortho_pairs[question]  # delete the old ones


decals_dr8_only_pairs = decals_pairs.copy()
decals_dr8_only_label_cols = decals_label_cols.copy()
decals_dr8_only_questions = decals_questions.copy()
decals_dr8_ortho_questions, decals_dr8_ortho_label_cols = schemas.extract_questions_and_label_cols(decals_dr8_ortho_pairs)

# the schema is slightly different for dr1/2 vs dr5+
# merging answers were changed completely
# bulge sizes were changed from 3 to 5
# bar was changed from yes/no to strong/weak/none
# spiral had 'cant tell' added
# TODO had distribution shifts despite the question not being changed (likely due to new icons on the website)
# the schema below is for training on dr1/2 only (not dr5+)
decals_dr12_pairs = {
    'smooth-or-featured': ['_smooth', '_featured-or-disk', '_artifact'],
    'disk-edge-on': ['_yes', '_no'],
    'has-spiral-arms': ['_yes', '_no'],
    'bar-dr12': ['_yes', '_no'],
    'bulge-size-dr12': ['_dominant', '_obvious', '_none'],
    'how-rounded-dr12': ['_completely', '_in-between', '_cigar-shaped'],  # completely was renamed to round
    'edge-on-bulge': ['_boxy', '_none', '_rounded'],
    'spiral-winding': ['_tight', '_medium', '_loose'],
    'spiral-arm-count-dr12': ['_1', '_2', '_3', '_4', '_more-than-4'],
    'merging-dr12': ['_neither', '_tidal-debris', '_both', '_merger']
}
decals_dr12_questions, decals_dr12_label_cols = schemas.extract_questions_and_label_cols(decals_dr12_pairs)

decals_dr12_only_pairs = {
    'smooth-or-featured': ['_smooth', '_featured-or-disk', '_artifact'],
    'disk-edge-on': ['_yes', '_no'],
    'has-spiral-arms': ['_yes', '_no'],
    'bar': ['_yes', '_no'],
    'bulge-size': ['_dominant', '_obvious', '_none'],
    'how-rounded': ['_completely', '_in-between', '_cigar-shaped'],  # completely was renamed to round
    'edge-on-bulge': ['_boxy', '_none', '_rounded'],
    'spiral-winding': ['_tight', '_medium', '_loose'],
    'spiral-arm-count': ['_1', '_2', '_3', '_4', '_more-than-4'],
    'merging': ['_neither', '_tidal-debris', '_both', '_merger']
}
decals_dr12_only_questions, decals_dr12_only_label_cols = schemas.extract_questions_and_label_cols(decals_dr12_only_pairs)

decals_dr12_ortho_pairs = {
    'smooth-or-featured-dr12': ['_smooth', '_featured-or-disk', '_artifact'],
    'disk-edge-on-dr12': ['_yes', '_no'],
    'has-spiral-arms-dr12': ['_yes', '_no'],
    'bar-dr12': ['_yes', '_no'],
    'bulge-size-dr12': ['_dominant', '_obvious', '_none'],
    'how-rounded-dr12': ['_completely', '_in-between', '_cigar-shaped'],  # completely was renamed to round
    'edge-on-bulge-dr12': ['_boxy', '_none', '_rounded'],
    'spiral-winding-dr12': ['_tight', '_medium', '_loose'],
    'spiral-arm-count-dr12': ['_1', '_2', '_3', '_4', '_more-than-4'],
    'merging-dr12': ['_neither', '_tidal-debris', '_both', '_merger']
}
decals_dr12_ortho_questions, decals_dr12_ortho_label_cols = schemas.extract_questions_and_label_cols(decals_dr12_ortho_pairs)

# I think performance should be best when training on *both*
# I made a joint dr1/2/5/8 catalog with columns drawn from all campaigns and shared where possible (dr5 and dr8 line up perfectly once a few dr5 w/ the old merger q are dropped)
decals_all_campaigns_pairs = {
    'smooth-or-featured': ['_smooth', '_featured-or-disk', '_artifact'],
    'disk-edge-on': ['_yes', '_no'],
    'has-spiral-arms': ['_yes', '_no'],
    'bar': ['_strong', '_weak', '_no'],
    'bulge-size': ['_dominant', '_large', '_moderate', '_small', '_none'],
    'how-rounded': ['_round', '_in-between', '_cigar-shaped'],
    'edge-on-bulge': ['_boxy', '_none', '_rounded'],
    'spiral-winding': ['_tight', '_medium', '_loose'],
    'spiral-arm-count': ['_1', '_2', '_3', '_4', '_more-than-4', '_cant-tell'],
    'merging': ['_none', '_minor-disturbance', '_major-disturbance', '_merger'],
    # and the dr1/2 versions which changed, separately
    'bar-dr12': ['_yes', '_no'],
    'bulge-size-dr12': ['_dominant', '_obvious', '_none'],
    'how-rounded-dr12': ['_completely', '_in-between', '_cigar-shaped'],
    'spiral-arm-count-dr12': ['_1', '_2', '_3', '_4', '_more-than-4'],
    'merging-dr12': ['_neither', '_tidal-debris', '_both', '_merger']
}
decals_all_campaigns_questions, decals_all_campaigns_label_cols = schemas.extract_questions_and_label_cols(decals_all_campaigns_pairs)

# very useful
decals_all_campaigns_ortho_pairs = {}
decals_all_campaigns_ortho_pairs.update(decals_dr12_ortho_pairs)
decals_all_campaigns_ortho_pairs.update(decals_dr5_ortho_pairs)
decals_all_campaigns_ortho_pairs.update(decals_dr8_ortho_pairs)
decals_all_campaigns_ortho_questions, decals_all_campaigns_ortho_label_cols = schemas.extract_questions_and_label_cols(decals_all_campaigns_ortho_pairs)


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
        'how-rounded': 'smooth-or-featured_smooth',
        'bulge-shape': 'disk-edge-on_yes',  # gz2 only
        'edge-on-bulge': 'disk-edge-on_yes',
        'spiral-winding': 'has-spiral-arms_yes',
        'spiral-arm-count': 'has-spiral-arms_yes', # bad naming...
        'merging': None,
        # 'something-odd': None,  # always asked
        # and the dr12 pairs (it's okay to include these in the dict, they'll simply be ignored if not keyed)
        'bar-dr12': featured_branch_answer,
        'bulge-size-dr12': featured_branch_answer,
        'how-rounded-dr12': 'smooth-or-featured_smooth',
        'spiral-arm-count-dr12': 'has-spiral-arms_yes',
        'merging-dr12': None
    }
    return dependencies


def get_decals_ortho_dependencies(question_answer_pairs):  # TODO remove arg
    """
    Get dict mapping each question (e.g. disk-edge-on) to the answer on which it depends (e.g. smooth-or-featured_featured-or-disk)

    Args:
        question_answer_pairs (dict): dict mapping questions (e.g. disk-edge-on) to list of answers (e.g. [disk-edge-on_yes, disk-edge-on_no])

    Returns:
        dict: see above
    """

    # luckily, these are the same in GZ2 and decals, just only some questions are asked
    dependencies = {
        # dr12
        'smooth-or-featured-dr12': None,
        'disk-edge-on-dr12': 'smooth-or-featured-dr12_featured-or-disk',
        'has-spiral-arms-dr12': 'disk-edge-on-dr12_no',
        'bar-dr12': 'disk-edge-on-dr12_no',
        'bulge-size-dr12': 'disk-edge-on-dr12_no',
        'how-rounded-dr12': 'smooth-or-featured-dr12_smooth',
        'edge-on-bulge-dr12': 'disk-edge-on-dr12_yes',
        'spiral-winding-dr12': 'has-spiral-arms-dr12_yes',
        'spiral-arm-count-dr12': 'has-spiral-arms-dr12_yes',
        'merging-dr12': None,
        # dr5
        'smooth-or-featured-dr5': None,  # always asked
        'disk-edge-on-dr5': 'smooth-or-featured-dr5_featured-or-disk',
        'has-spiral-arms-dr5': 'disk-edge-on-dr5_no',
        'bar-dr5': 'disk-edge-on-dr5_no',
        'bulge-size-dr5': 'disk-edge-on-dr5_no',
        'how-rounded-dr5': 'smooth-or-featured-dr5_smooth',
        'edge-on-bulge-dr5': 'disk-edge-on-dr5_yes',
        'spiral-winding-dr5': 'has-spiral-arms-dr5_yes',
        'spiral-arm-count-dr5': 'has-spiral-arms-dr5_yes', # bad naming...
        'merging-dr5': None,
        # dr8 is identical to dr5, just with -dr8
        'smooth-or-featured-dr8': None,
        'disk-edge-on-dr8': 'smooth-or-featured-dr8_featured-or-disk',
        'has-spiral-arms-dr8': 'disk-edge-on-dr8_no',
        'bar-dr8': 'disk-edge-on-dr8_no',
        'bulge-size-dr8': 'disk-edge-on-dr8_no',
        'how-rounded-dr8': 'smooth-or-featured-dr8_smooth',
        'edge-on-bulge-dr8': 'disk-edge-on-dr8_yes',
        'spiral-winding-dr8': 'has-spiral-arms-dr8_yes',
        'spiral-arm-count-dr8': 'has-spiral-arms-dr8_yes',
        'merging-dr8': None,
    }
    return dependencies

rings_pairs = {
    'ring': ['_yes', '_no']
}
rings_questions, rings_label_cols = schemas.extract_questions_and_label_cols(rings_pairs)
def rings_dependencies():
    return {'ring': None}


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

