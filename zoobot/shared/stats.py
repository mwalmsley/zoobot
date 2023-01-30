import numpy as np
from scipy.stats import beta

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


# TODO for q in...
# supports trailing dimensions for more distributions
def beta_mixture_on_grid(concentrations, question_indices, answer_index, gridsize=100):
    # concentration (galaxy, answer_index, (distribution dims))

    # flatten trailing dists
    concentrations = concentrations.reshape(concentrations.shape[0], concentrations.shape[1], -1)
    # (galaxy, answer_index, distribution)

    # reshape to have distribution in leading dim
    concentrations = concentrations.transpose(2, 0, 1)
    # (distribution, galaxy, answer_index)
    # print(concentrations.shape)
    # exit()

    concentrations_q = concentrations[:, :, question_indices[0]:question_indices[1]+1]
    concentrations_a = concentrations[:, :, answer_index]
    concentrations_sum = concentrations_q.sum(axis=2)
    # dirichlet of this or not this is equivalent to beta distribution with concentrations (this, sum_of_not_this)
    concentrations_not_a = concentrations_sum - concentrations_a

    # print(concentrations_a.shape, concentrations_not_a.shape)
    # (distribution, galaxy)

    dist_with_extra_dim = beta(a=np.expand_dims(concentrations_a, -1), b=np.expand_dims(concentrations_not_a, -1))

    grid = np.linspace(1e-8, 1. - 1e-8, num=gridsize).reshape(1, -1)  # galaxy, test_cdf_value
    pdf_grid = dist_with_extra_dim.pdf(grid)
    # normalise over grid dimension
    pdf_grid = pdf_grid / np.sum(pdf_grid, axis=2, keepdims=True)
    print(pdf_grid.shape)
    # print(pdf_grid.shape) # (distribution, galaxy, grid) 

    # take mean over distribution dim (0th)
    mean_pdf = pdf_grid.mean(axis=0)
    # (galaxy, grid)

    # convert to cdf via rolling sum over grid axis
    cdf = np.cumsum(mean_pdf, axis=1)

    return grid.squeeze(), mean_pdf, cdf


def get_confidence_interval(grid, pdf, cdf, interval_width=.95):  # grid (100), pdf/cdf (4, 100)

    expected_value = np.mean(grid * pdf, axis=1)  # (4)
    # print(expected_value.shape)
    # print(expected_value)

    loc_of_expected = np.argmin(np.abs(grid.reshape(1, -1) - expected_value.reshape(-1, 1)), axis=1)
    # print(loc_of_expected)
    # print(loc_of_expected.shape)  (galaxy), values of grid_loc (aimed at grid dimension)

    # print(cdf.shape)  # galaxy, grid
    # transpose to slice cdf by grid_loc values
    # then take diagonal for first grid_loc value being for first galaxy, ...
    cdf_at_loc_of_expected = cdf.T[loc_of_expected].diagonal()

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

    return grid.squeeze()[index_of_lower], grid.squeeze()[index_of_higher]





def test_beta_cdf_on_grid():
    question_indices = [0, 2]
    answer_index = 1
    concentrations = np.expand_dims(np.array([[2., 2., 4.], [4., 4., 2.], [4., 4., 2.], [4., 4., 2.]]), axis=2)  # (galaxy=4, answer=3, distribution=1)
    # print(concentrations.shape)
    # exit()

    grid, pdf, cdf = beta_mixture_on_grid(concentrations, question_indices, answer_index, gridsize=1000)
    print(grid.shape, pdf.shape, cdf.shape)

    lower_fracs, upper_fracs = get_confidence_interval(grid, pdf, cdf)

    with np.printoptions(precision=3):
        print(lower_fracs)
        print(upper_fracs)


# def test_get_confidence_interval_around_mode():

#     question_indices = [0, 2]
#     answer_index = 1
#     concentrations = np.array([[2., 2., 4.], [4., 4., 2.]])  # 2 events/galaxies, 3 possible outcomes/answers
#     # should collapse to [[2., 6.], [4., 6.]] in beta [this, not-this] for answer 1

#     # https://homepage.divms.uiowa.edu/~mbognar/applets/beta.html
#     # mode 0.25, 0.4
#     # if lower bound 0: upper bound 0.5207, 0.65506
#     # works well for large gridsize (>100)


#     lower_fracs, upper_fracs = get_confidence_interval_around_mode(concentrations, question_indices, answer_index, gridsize=100)

#     with np.printoptions(precision=3):
#         print(lower_fracs)
#         print(upper_fracs)


if __name__ == '__main__':

    test_beta_cdf_on_grid()