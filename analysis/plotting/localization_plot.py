import numpy
import scipy
from matplotlib import pyplot as plt
import slab

def localization_accuracy(sequence, show=True, plot_dim=2, binned=True, axis=None, show_single_responses=True,
                          elevation='all', azimuth='all'):
    if sequence.this_n == -1 or sequence.n_remaining == 132 or not sequence.data:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan
    # retrieve RCX_files
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    targets = loc_data[:, 1]  # [az, ele]
    responses = loc_data[:, 0]
    elevations = numpy.unique(loc_data[:, 1, 1])
    azimuths = numpy.unique(loc_data[:, 1, 0])
    targets[:, 1] = loc_data[:, 1, 1]  # target elevations
    responses[:, 1] = loc_data[:, 0, 1]  # percieved elevations
    targets[:, 0] = loc_data[:, 1, 0]
    responses[:, 0] = loc_data[:, 0, 0]
    #  elevation gain, rmse, response variability
    elevation_gain, n = scipy.stats.linregress(targets[:, 1], responses[:, 1])[:2]
    rmse = numpy.sqrt(numpy.mean(numpy.square(targets - responses), axis=0))
    variability = numpy.mean([numpy.std(responses[numpy.where(numpy.all(targets == target, axis=1))], axis=0)
                    for target in numpy.unique(targets, axis=0)], axis=0)

    az_rmse, ele_rmse = rmse[0], rmse[1]
    az_sd, ele_sd = variability[0], variability[1]

    # target ids
    right_ids = numpy.where(loc_data[:, 1, 0] > 0)
    left_ids = numpy.where(loc_data[:, 1, 0] < 0)
    mid_ids = numpy.where(loc_data[:, 1, 0] == 0)
    # above_ids = numpy.where(loc_data[:, 1, 1] > 0)
    # below_ids = numpy.where(loc_data[:, 1, 1] < 0)

    # mean perceived location for each target speaker
    i = 0
    mean_loc = numpy.zeros((45, 2, 2))
    for az_id, azimuth in enumerate(numpy.unique(targets[:, 0])):
        for ele_id, elevation in enumerate(numpy.unique(targets[:, 1])):
            [perceived_targets] = loc_data[numpy.where(numpy.logical_and(loc_data[:, 1, 1] == elevation,
                          loc_data[:, 1, 0] == azimuth)), 0]
            if perceived_targets.size != 0:
                mean_perceived = numpy.mean(perceived_targets, axis=0)
                mean_loc[i] = numpy.array(((azimuth, mean_perceived[0]), (elevation, mean_perceived[1])))
                i += 1

    # divide target space in 16 half overlapping sectors and get mean response for each sector
    mean_loc_binned = numpy.empty((0, 2, 2))
    for a in range(6):
        for e in range(6):
            tar_bin = loc_data[numpy.logical_or(loc_data[:, 1, 0] == azimuths[a],
                                                loc_data[:, 1, 0] == azimuths[a+1])]
            tar_bin = tar_bin[numpy.logical_or(tar_bin[:, 1, 1] == elevations[e],
                                               tar_bin[:, 1, 1] == elevations[e+1])]
            tar_bin[:, 1] = numpy.array((numpy.mean([azimuths[a], azimuths[a+1]]),
                                         numpy.mean([elevations[e], elevations[e+1]])))
            mean_tar_bin = numpy.mean(tar_bin, axis=0).T
            mean_tar_bin[:, [0, 1]] = mean_tar_bin[:, [1, 0]]
            mean_loc_binned = numpy.concatenate((mean_loc_binned, [mean_tar_bin]))

    if show:
        if not axis:
            fig, axis = plt.subplots(1, 1)
        elevation_ticks = numpy.unique(targets[:, 1])
        azimuth_ticks = numpy.unique(targets[:, 0])
        # axis.set_yticks(elevation_ticks)
        # axis.set_ylim(numpy.min(elevation_ticks)-15, numpy.max(elevation_ticks)+15)
        if plot_dim == 2:
            # axis.set_xticks(azimuth_ticks)
            # axis.set_xlim(numpy.min(azimuth_ticks)-15, numpy.max(azimuth_ticks)+15)
            if show_single_responses:
                axis.scatter(responses[:, 0], responses[:, 1], s=8, edgecolor='grey', facecolor='none')
            if binned:
                azimuths = numpy.unique(mean_loc_binned[:, 0, 0])
                elevations = numpy.unique(mean_loc_binned[:, 1, 0])
                mean_loc = mean_loc_binned
                # azimuth_ticks = azimuths
                # elevation_ticks = elevations
            for az in azimuths:  # plot lines between target locations
                [x] = mean_loc[numpy.where(mean_loc[:, 0, 0] == az), 0, 0]
                [y] = mean_loc[numpy.where(mean_loc[:, 0, 0] == az), 1, 0]
                axis.plot(x, y, color='black', linewidth=0.5)
            for ele in elevations:
                [x] = mean_loc[numpy.where(mean_loc[:, 1, 0] == ele), 0, 0]
                [y] = mean_loc[numpy.where(mean_loc[:, 1, 0] == ele), 1, 0]
                axis.plot(x, y, color='black', linewidth=0.5)
            axis.scatter(mean_loc[:, 0, 1], mean_loc[:, 1, 1], color='black', s=25)
            for az in azimuths:  # plot lines between mean perceived locations for each target
                [x] = mean_loc[numpy.where(mean_loc[:, 0, 0] == az), 0, 1]
                [y] = mean_loc[numpy.where(mean_loc[:, 0, 0] == az), 1, 1]
                axis.plot(x, y, color='black')
            for ele in elevations:
                [x] = mean_loc[numpy.where(mean_loc[:, 1, 0] == ele), 0, 1]
                [y] = mean_loc[numpy.where(mean_loc[:, 1, 0] == ele), 1, 1]
                axis.plot(x, y, color='black')
        elif plot_dim == 1:
            axis.set_xticks(elevation_ticks)
            axis.set_xlim(numpy.min(elevation_ticks)-15, numpy.max(elevation_ticks)+15)
            axis.set_xlabel('target elevations')
            axis.set_ylabel('perceived elevations')
            axis.grid(visible=True, which='major', axis='both', linestyle='dashed', linewidth=0.5, color='grey')
            axis.set_axisbelow(True)
            # scatter plot with regression line (elevation gain)
            axis.scatter(targets[:, 1][left_ids], responses[:, 1][left_ids], s=10, c='red', label='left')
            axis.scatter(targets[:, 1][right_ids], responses[:, 1][right_ids], s=10, c='blue', label='right')
            axis.scatter(targets[:, 1][mid_ids], responses[:, 1][mid_ids], s=10, c='black', label='middle')
            x = numpy.arange(-55, 56)
            y = elevation_gain * x + n
            axis.plot(x, y, c='grey', linewidth=1, label='elevation gain %.2f' % elevation_gain)
            plt.legend()
        axis.set_yticks(elevation_ticks)
        axis.set_ylim(numpy.min(elevation_ticks) - 15, numpy.max(elevation_ticks) + 15)
        axis.set_xticks(azimuth_ticks)
        axis.set_xlim(numpy.min(azimuth_ticks) - 15, numpy.max(azimuth_ticks) + 15)
        axis.set_title('elevation gain: %.2f' % elevation_gain)
        plt.show()
    #  return EG, RMSE and Response Variability
    return elevation_gain, ele_rmse, ele_sd, az_rmse, az_sd


def load_latest(subject_dir):
    file_list = []
    for file_name in sorted(list(subject_dir.iterdir())):
        if file_name.is_file() and file_name.suffix != '.sofa':
            file_list.append(file_name)
            file_list.sort()
    sequence = slab.Trialsequence(conditions=45, n_reps=3)
    sequence.load_pickle(file_name=file_list[-1])
    print(f'Loaded {file_list[-1].name}')
    return(sequence)


def get_target_proabilities(sequence, show=False, axis=None):
    # calculate target probabilities depending on localization error
    # retrieve data
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    responses = loc_data[:, 0]
    azimuths = numpy.unique(loc_data[:, 1, 0])
    elevations = numpy.unique(loc_data[:, 1, 1])
    targets = []
    for az in azimuths:  # this makes no sense but is in alignment with ff speaker table
        for ele in -numpy.sort(-elevations):
            targets.append([az, ele])
    targets = numpy.asarray(targets)
    targets = numpy.delete(targets, [0, 6, 24, -1, -7], axis=0)
    # targets = numpy.unique(loc_data[:, 1], axis=0)
    # mean response error each target speaker
    response_error = numpy.zeros((len(targets), 3))
    for idx, target in enumerate(targets):
        [perceived_targets] = loc_data[numpy.where(numpy.all(loc_data[:, 1] == target, axis=1)), 0]
        mean_response = numpy.mean(perceived_targets, axis=0)
        error = numpy.linalg.norm(target - mean_response)
        response_error[idx] = numpy.append(target, error)
    target_p = numpy.expand_dims(response_error[:, 2], axis=1) / numpy.sum(response_error[:, 2])
    response_error = numpy.hstack((response_error, target_p))
    if show:
        elevations = numpy.unique(loc_data[:, 1, 1])
        azimuths = numpy.unique(loc_data[:, 1, 0])
        img = numpy.zeros((len(elevations), len(azimuths)))
        for target in targets:
            az_idx = numpy.where(azimuths == target[0])[0][0]
            ele_idx = numpy.where(elevations == target[1])[0][0]
            img[ele_idx][az_idx] = response_error[numpy.all(response_error[:, :2] == target, axis=1), 3]
        img[img == 0] = None
        if not axis:
            fig, axis = plt.subplots()
        cbar_levels = numpy.linspace(0, 0.1, 10)
        contour = axis.pcolormesh(azimuths, elevations, img)
        cax_pos = list(axis.get_position().bounds)  # (x0, y0, width, height)
        cax_pos[0] += 0.8  # x0
        cax_pos[2] = 0.012  # width
        cax = fig.add_axes(cax_pos)
        cbar = fig.colorbar(contour, cax, orientation="vertical", ticks=numpy.linspace(0, 0.1, 5))
        cax.set_title('Probability')
        axis.set_xlabel('Azimuth')
        axis.set_ylabel('Elevation')
        axis.set_yticks(elevations)
        axis.set_ylim(numpy.min(elevations) - 15, numpy.max(elevations) + 15)
        axis.set_xticks(azimuths)
        axis.set_xlim(numpy.min(azimuths) - 15, numpy.max(azimuths) + 15)
        localization_accuracy(sequence, show=True, plot_dim=2, binned=True)
    return response_error