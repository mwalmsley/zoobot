

def extract_questions_and_label_cols(question_answer_pairs):
    """
    Convenience wrapper to get list of questions and label_cols from a schema.
    Common starting point for analysis, iterating over questions, etc.

    Args:
        question_answer_pairs (dict): e.g. {'smooth-or-featured: ['_smooth, _featured-or-disk, ...], ...}

    Returns:
        list: all questions e.g. [Question('smooth-or-featured'), ...]
        list: label_cols (list of answer strings). See ``label_metadata.py`` for examples.
    """
    questions = list(question_answer_pairs.keys())
    label_cols = [q + answer for q, answers in question_answer_pairs.items() for answer in answers]
    return questions, label_cols


# use these by importing them in another script e.g.
# from zoobot.label_metadata import decals_label_cols 

# dr5, implicitly (deprecated - use decals_dr5_ortho_pairs below instead, for clarity)
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
decals_dr5_ortho_questions, decals_dr5_ortho_label_cols = extract_questions_and_label_cols(decals_dr5_ortho_pairs)

# exactly the same for dr8. 
decals_dr8_ortho_pairs = decals_dr5_ortho_pairs.copy()
for question, answers in decals_dr8_ortho_pairs.copy().items(): # avoid modifying while looping
    decals_dr8_ortho_pairs[question.replace('-dr5', '-dr8')] = answers
    del decals_dr8_ortho_pairs[question]  # delete the old ones


decals_dr8_only_pairs = decals_pairs.copy()
decals_dr8_only_label_cols = decals_label_cols.copy()
decals_dr8_only_questions = decals_questions.copy()
decals_dr8_ortho_questions, decals_dr8_ortho_label_cols = extract_questions_and_label_cols(decals_dr8_ortho_pairs)

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
decals_dr12_questions, decals_dr12_label_cols = extract_questions_and_label_cols(decals_dr12_pairs)

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
decals_dr12_only_questions, decals_dr12_only_label_cols = extract_questions_and_label_cols(decals_dr12_only_pairs)

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
decals_dr12_ortho_questions, decals_dr12_ortho_label_cols = extract_questions_and_label_cols(decals_dr12_ortho_pairs)

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
decals_all_campaigns_questions, decals_all_campaigns_label_cols = extract_questions_and_label_cols(decals_all_campaigns_pairs)

# very useful
decals_all_campaigns_ortho_pairs = {}
decals_all_campaigns_ortho_pairs.update(decals_dr12_ortho_pairs)
decals_all_campaigns_ortho_pairs.update(decals_dr5_ortho_pairs)
decals_all_campaigns_ortho_pairs.update(decals_dr8_ortho_pairs)
decals_all_campaigns_ortho_questions, decals_all_campaigns_ortho_label_cols = extract_questions_and_label_cols(decals_all_campaigns_ortho_pairs)


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
gz2_partial_pairs = {
    'smooth-or-featured': ['_smooth', '_featured-or-disk']
}
gz2_partial_questions, gz2_partial_label_cols = extract_questions_and_label_cols(gz2_partial_pairs)


"""
Dict mapping each question (e.g. disk-edge-on) to the answer on which it depends (e.g. smooth-or-featured_featured-or-disk)
"""
# deprecated - use decals_ortho_dependencies below instead for clarity
gz2_and_decals_dependencies = {
    'smooth-or-featured': None,  # always asked
    'disk-edge-on': 'smooth-or-featured_featured-or-disk',
    'has-spiral-arms': 'smooth-or-featured_featured-or-disk',
    'bar': 'smooth-or-featured_featured-or-disk',
    'bulge-size': 'smooth-or-featured_featured-or-disk',
    'how-rounded': 'smooth-or-featured_smooth',
    'bulge-shape': 'disk-edge-on_yes',  # gz2 only
    'edge-on-bulge': 'disk-edge-on_yes',
    'spiral-winding': 'has-spiral-arms_yes',
    'spiral-arm-count': 'has-spiral-arms_yes', # bad naming...
    'merging': None,
    # 'something-odd': None,  # always asked
    # and the dr12 pairs (it's okay to include these in the dict, they'll simply be ignored if not keyed)
    'bar-dr12': 'smooth-or-featured_featured-or-disk',
    'bulge-size-dr12': 'smooth-or-featured_featured-or-disk',
    'how-rounded-dr12': 'smooth-or-featured_smooth',
    'spiral-arm-count-dr12': 'has-spiral-arms_yes',
    'merging-dr12': None,
    # and gz2
    'something-odd': None
}


decals_ortho_dependencies = {
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


rings_pairs = {
    'ring': ['_yes', '_no']
}
rings_questions, rings_label_cols = extract_questions_and_label_cols(rings_pairs)

rings_dependencies = {'ring': None }
