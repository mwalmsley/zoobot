import numpy as np
from scipy.stats import beta


def expected_value_of_dirichlet_mixture(concentrations, schema):
    # badly named vs posteriors, actually gives predicted vote fractions of answers...
    # mean probability (including dropout) of an answer being given. 
    # concentrations has (batch, answer, ...) shape where extra dims are distributions (e.g. galaxies, answer, model, pass)

    # collapse trailing dims
    concentrations = concentrations.reshape(concentrations.shape[0], concentrations.shape[1], -1)

    p_of_answers = []
    for q in schema.questions:
        concentrations_by_q = concentrations[:, q.start_index:q.end_index+1]  # trailing dims are distributions
        for answer_index in range(len(q.answers)):
            # first mean is per distribution
            # second .mean() is average p for all distributions
            mean_of_each_distribution = get_beta_mean(concentrations=concentrations_by_q, answer_index=answer_index)  # (galaxies, distributions)
            # print(mean_of_each_distribution.shape)
            p_of_answers.append(mean_of_each_distribution.mean(axis=1))  # now (galaxies)

    p_of_answers = np.stack(p_of_answers, axis=1)
    return p_of_answers


def get_beta_mean(concentrations, answer_index):
    # concentrations shape (galaxy, answer, distribution)
    concentrations_a = concentrations[:, answer_index]
    concentrations_sum = concentrations.sum(axis=1)
    return concentrations_a / concentrations_sum   # (galaxy, distribution)



# Prob given q is asked
# get prob_of_answers = expected_value_of_dirichlet_mixture(concentrations, schema)
def get_expected_votes_ml(prob_of_answers, question, votes_for_base_question: int, schema, round_votes):
    # (send all prob_of_answers not per-question prob, they are all potentially relevant)
    prev_q = question.asked_after
    if prev_q is None:
        expected_votes = np.ones(len(prob_of_answers)) * votes_for_base_question
    else:
        joint_p_of_asked = schema.joint_p(prob_of_answers, prev_q.text)  # prob of getting the answer needed to ask this question
        expected_votes = joint_p_of_asked * votes_for_base_question
    if round_votes:
        return np.round(expected_votes)
    else:
        return expected_votes


def get_expected_votes_human(label_df, question, votes_for_base_question: int, schema, round_votes):
    # might be better called "get galaxies where at least half of humans actually answered that question, in the labels"
    

    answer_fractions = label_df[[a + '_fraction' for a in schema.label_cols]].values
    # some will be nan as fractions are often nan (as 0 of 0)

    prev_q = question.asked_after
    if prev_q is None:
        expected_votes = np.ones(len(label_df)) * votes_for_base_question
    else:
        # prob of getting the answer needed to ask this question - the product of the (perhaps expected, but here, actual) dependent vote fractions
        joint_p_of_asked = schema.joint_p(answer_fractions, prev_q.text)  
        # for humans, its just votes_for_base_question * the product of all the fractions leading to that q
        expected_votes = joint_p_of_asked * votes_for_base_question
    if round_votes:
        return np.round(expected_votes)
    else:
        return expected_votes




# class DirichletEqualMixture():

#     def __init__(self, concentrations) -> None:
        
#         # concentrations should be shape (galaxy, answer, distributions...)

#         

#         self._concentrations = concentrations

#         # create stats object to do all the work
#         self._dist = dirichlet(self._concentrations)

#     def pdf(self, x):
#         return self._dist.pdf(x)

#     def mean(self):
#         return self._dist.mean()



# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html


# for single distribution/model
# def get_confidence_interval_around_mode(concentrations, question_indices, answer_index, interval_width=0.95, gridsize=20):
    
#     concentrations_q = concentrations[:, question_indices[0]:question_indices[1]+1]
#     concentrations_a = concentrations[:, answer_index]
#     concentrations_sum = concentrations_q.sum(axis=1)
#     # dirichlet of this or not this is equivalent to beta distribution with concentrations (this, sum_of_not_this)
#     concentrations_not_a = concentrations_sum - concentrations_a
#     # concentrations_a and concentrations_not_a have shape (batch)

#     # find mode
#     mode = (concentrations_a - 1) / (concentrations_a + concentrations_sum - 2)  # (10,)
#     # can broadcast in galaxy/batch dimension by default!
#     dist = beta(a=concentrations_a, b=concentrations_not_a)
#     cdf_at_mode = dist.cdf(mode)

#     # find what cdf value corresponds to 0.95 width interval around the mode
#     lower_cdf_value = cdf_at_mode - interval_width/2.
#     upper_cdf_value = cdf_at_mode + interval_width/2.

#     # account for edges - replace other bound with interval_width
#     # if upper cdf value would have to be above 1 to include 0.95 mass, set lower bound to e.g. cdf of 0.05
#     upper_bound_above_1 = upper_cdf_value >= 1.
#     lower_cdf_value = np.where(upper_bound_above_1, 1-interval_width, lower_cdf_value)
#     # and set upper bound to exactly 1 / index -1 if needed, below

#     # similarly for lower bound - set upper bound to e.g. cdf of 0.95
#     lower_bound_below_0 = lower_cdf_value <= 0.
#     upper_cdf_value = np.where(lower_bound_below_0, interval_width, upper_cdf_value)

#     # find what x points (vote fractions) have those cdf values
#     # can broadcast in bound dimension with a reshape!
#     dist_with_extra_dim = beta(a=concentrations_a.reshape(-1, 1), b=concentrations_not_a.reshape(-1, 1))
#     grid = np.linspace(0., 1., num=gridsize).reshape(1, -1)  # galaxy, test_cdf_value
#     cdf_grid = dist_with_extra_dim.cdf(grid)
#     # now work out which indices have those values
#     index_of_lower = np.argmin(np.abs(cdf_grid - lower_cdf_value.reshape(-1, 1)), axis=1)
#     index_of_higher = np.argmin(np.abs(cdf_grid - upper_cdf_value.reshape(-1, 1)), axis=1)

#     return grid.squeeze()[index_of_lower], grid.squeeze()[index_of_higher]


def get_confidence_intervals(concentrations, schema, interval_width=.9, gridsize=100):

    # collapse trailing dims
    concentrations = concentrations.reshape(concentrations.shape[0], concentrations.shape[1], -1)

    lower_edges = []
    upper_edges = []
    for q in schema.questions:
        concentrations_q = concentrations[:, q.start_index:q.end_index+1]
        for answer_index in range(len(q.answers)):
            # grid, pdf, cdf = beta_mixture_on_grid(concentrations_q, answer_index, gridsize=gridsize)
            # lower_edge, upper_edge = get_confidence_interval_from_binned_dist(grid, pdf, cdf, interval_width=interval_width)
            lower_edge, upper_edge = get_confidence_interval_from_ppf_medians(concentrations_q, answer_index, interval_width=interval_width)
            lower_edges.append(lower_edge)
            upper_edges.append(upper_edge)

    lower_edges = np.stack(lower_edges, axis=1)
    upper_edges = np.stack(upper_edges, axis=1)

    return lower_edges, upper_edges


def get_confidence_interval_from_ppf_medians(concentrations_q, answer_index, interval_width=.9):
    concentrations_a, concentrations_not_a = reshape_concentrations_for_scipy_beta(concentrations_q, answer_index)
    dist = beta(a=concentrations_a, b=concentrations_not_a)
    lower_edges = dist.ppf(0.5 - interval_width/2.)  # dimension (distribution, galaxy)
    upper_edges = dist.ppf(0.5 + interval_width/2.)
    return np.median(lower_edges, axis=0), np.median(upper_edges, axis=0)  # median over mixture, per galaxy



# supports trailing dimensions for more distributions
def beta_mixture_on_grid(concentrations_q, answer_index, gridsize=100):
    # concentration (galaxy, answer_index, distribution), any extra distribution dims already flattened

    concentrations_a, concentrations_not_a = reshape_concentrations_for_scipy_beta(concentrations_q, answer_index)
    # (distribution, galaxy)

    dist_with_extra_dim = beta(a=np.expand_dims(concentrations_a, -1), b=np.expand_dims(concentrations_not_a, -1))

    grid = np.linspace(1e-8, 1. - 1e-8, num=gridsize).reshape(1, -1)  # galaxy, test_cdf_value
    pdf_grid = dist_with_extra_dim.pdf(grid)
    # normalise over grid dimension
    pdf_grid = pdf_grid / np.sum(pdf_grid, axis=2, keepdims=True)
    # (distribution, galaxy, grid) 

    # take mean over distribution dim (0th)
    mean_pdf = pdf_grid.mean(axis=0)
    # (galaxy, grid)

    # convert to cdf via rolling sum over grid axis
    cdf = np.cumsum(mean_pdf, axis=1)

    return grid.squeeze(), mean_pdf, cdf


def get_confidence_interval_from_binned_dist(grid, pdf, cdf, interval_width=.95):  # grid (100), pdf/cdf (4, 100)

    expected_value = np.sum(grid.reshape(1, -1) * pdf, axis=1)  # (galaxies)
    # print(expected_value.shape)

    loc_of_expected = np.argmin(np.abs(grid.reshape(1, -1) - expected_value.reshape(-1, 1)), axis=1)
    # (galaxy), values of grid_loc (aimed at grid dimension)

    # iterating through the ith element of loc_of_expected, k, we want to select [i, k] from cdf
    # can do this with multi-dim integer array indexing
    # https://numpy.org/devdocs/user/basics.indexing.html#integer-array-indexing
    grid_indices = np.arange(len(loc_of_expected)), loc_of_expected   # NOT a list, or broadcasts to extra dim
    cdf_at_loc_of_expected = cdf[grid_indices]

    lower_cdf_value = cdf_at_loc_of_expected - interval_width/2.
    upper_cdf_value = cdf_at_loc_of_expected + interval_width/2.

    # account for edges - replace other bound with interval_width
    # if upper cdf value would have to be above 1 to include 0.95 mass, set lower bound to e.g. cdf of 0.05
    upper_bound_above_1 = upper_cdf_value >= 1.
    lower_cdf_value = np.where(upper_bound_above_1, 1-interval_width, lower_cdf_value)
    # and set upper bound to exactly 1 / index -1 if needed, below

    # similarly for lower bound - set upper bound to e.g. cdf of 0.95
    lower_bound_below_0 = lower_cdf_value <= 0.
    upper_cdf_value = np.where(lower_bound_below_0, interval_width, upper_cdf_value)

    # now work out which indices have those values
    index_of_lower = np.argmin(np.abs(cdf - lower_cdf_value.reshape(-1, 1)), axis=1)
    index_of_higher = np.argmin(np.abs(cdf - upper_cdf_value.reshape(-1, 1)), axis=1)

    # each of shape (galaxy)
    return grid.squeeze()[index_of_lower], grid.squeeze()[index_of_higher]



def reshape_concentrations_for_scipy_beta(concentrations_q, answer_index):
    # reshape to have distribution in leading dim
    concentrations_q = concentrations_q.transpose(2, 0, 1)
    # (distribution, galaxy, answer_index)

    concentrations_a = concentrations_q[:, :, answer_index]
    concentrations_q_sum = concentrations_q.sum(axis=2)
    # dirichlet of this or not this is equivalent to beta distribution with concentrations (this, sum_of_not_this)
    concentrations_not_a = concentrations_q_sum - concentrations_a

    return concentrations_a, concentrations_not_a



def test_beta_cdf_on_grid():
    question_indices = [0, 2]
    answer_index = 1
    concentrations = np.expand_dims(np.array([[2., 2., 4.], [4., 4., 2.], [4., 4., 2.], [4., 4., 2.]]), axis=2)  # (galaxy=4, answer=3, distribution=1)


    grid, pdf, cdf = beta_mixture_on_grid(concentrations, question_indices, answer_index, gridsize=1000)
    print(grid.shape, pdf.shape, cdf.shape)

    lower_fracs, upper_fracs = get_confidence_interval_from_binned_dist(grid, pdf, cdf)

    with np.printoptions(precision=3):
        print(lower_fracs)
        print(upper_fracs)

#     question_indices = [0, 2]
#     answer_index = 1
#     concentrations = np.array([[2., 2., 4.], [4., 4., 2.]])  # 2 events/galaxies, 3 possible outcomes/answers
#     # should collapse to [[2., 6.], [4., 6.]] in beta [this, not-this] for answer 1

#     # https://homepage.divms.uiowa.edu/~mbognar/applets/beta.html
#     # mode 0.25, 0.4
#     # if lower bound 0: upper bound 0.5207, 0.65506
#     # works well for large gridsize (>100)

# def test_dirichlet_mixture():
#     concentrations = np.expand_dims(np.array([[2., 2., 4.], [4., 4., 2.], [4., 4., 2.], [4., 4., 2.]]), axis=2)

#     mixture = DirichletEqualMixture(concentrations)
#     print(mixture.mean())

def test_get_confidence_interval_from_ppf_medians():
    # question_indices = [0, 2]
    answer_index = 1
    concentrations = np.array([[2., 2., 4.], [4., 4., 2.]])
    # concentrations = np.expand_dims(concentrations, axis=2)
    concentrations = np.stack([concentrations] * 5, axis=2)
    print(concentrations.shape)
    lower_edge, upper_edge = get_confidence_interval_from_ppf_medians(concentrations, answer_index)
    print(lower_edge, upper_edge)


if __name__ == '__main__':

#     # test_beta_cdf_on_grid()

#     test_dirichlet_mixture()

    test_get_confidence_interval_from_ppf_medians()