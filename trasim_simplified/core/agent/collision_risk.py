'''
To compute two-dimensional Time-To-Collision,
use function TTC(samples, 'dataframe') or TTC(samples, 'values');

To compute two-dimensional Deceleration Rate to Avoid Collision,
use function DRAC(samples, 'dataframe') or DRAC(samples, 'values');

To compute two-dimensional Modified Time-To-Collision,
use function MTTC(samples, 'dataframe') or MTTC(samples, 'values');

To compute all the above three indicators at once,
use function TTC_DRAC_MTTC(samples, 'dataframe') or TTC_DRAC_MTTC(samples, 'values').

The first input `samples` is a pandas dataframe of vehicle pair samples, 
which should include the following columns:
------------------------------------------------------------------------------------------------------------
x_i      :  x coordinate of the ego vehicle (usually assumed to be centroid)                               |
y_i      :  y coordinate of the ego vehicle (usually assumed to be centroid)                               |
vx_i     :  x coordinate of the velocity of the ego vehicle                                                |
vy_i     :  y coordinate of the velocity of the ego vehicle                                                |
hx_i     :  x coordinate of the heading direction of the ego vehicle                                       |
hy_i     :  y coordinate of the heading direction of the ego vehicle                                       |
acc_i    :  acceleration along the heading direction of the ego vehicle (only required if computing MTTC)  |
length_i :  length of the ego vehicle                                                                      |
width_i  :  width of the ego vehicle                                                                       |
x_j      :  x coordinate of another vehicle (usually assumed to be centroid)                               |
y_j      :  y coordinate of another vehicle (usually assumed to be centroid)                               |
vx_j     :  x coordinate of the velocity of another vehicle                                                |
vy_j     :  y coordinate of the velocity of another vehicle                                                |
hx_j     :  x coordinate of the heading direction of another vehicle                                       |
hy_j     :  y coordinate of the heading direction of another vehicle                                       |
acc_j    :  acceleration along the heading direction of another vehicle (optional)                         |
length_j :  length of another vehicle                                                                      |
width_j  :  width of another vehicle                                                                       |
------------------------------------------------------------------------------------------------------------
The second input allows outputing 
    1) a dataframe with inputed samples plus new column(s) of the requested indicator, or
    2) a numpy array of the requested indicator values.

The ego vehicle and another vehicle will never collide if they keep current speed when 
    1) indicator==np.inf when indicator==TTC or indicator==MTTC, or
    2) indicator==0 when indicator==DRAC.

When indicator<0, the bounding boxes of the ego vehicle and another vehicle are overlapping.
This is due to approximating the space occupied by a vehicle with a rectangular.
In other words, negative indicator in this computation means the collision between the two 
vehicles almost (or although seldom, already) occurred.

*** Note that the computation can return extreme small positive values (for TTC/MTTC) or 
    extreme large values (for DRAC) even when the vehivles overlap a bit (so should be negative values). 
    In order to improve the accuracy, please use function CurrentD(samples, 'dataframe') or 
    CurrentD(samples, 'values') to further exclude overlapping vehicles.

######################### Copyright (c) 2025 Yiru Jiao <y.jiao-1@tudelft.nl> ###########################
'''

# Import
import numpy as np
import warnings

import pandas as pd
from scipy.stats import multivariate_normal

from trasim_simplified.core.constant import TrackInfo as TI, TrajPoint


# Useful functions
def line(point0, point1):
    """
    Get the line equation from two points.
    """
    x0, y0 = point0
    x1, y1 = point1
    a = y0 - y1
    b = x1 - x0
    c = x0 * y1 - x1 * y0
    return a, b, c


def intersect(line0, line1):
    """
    Get the intersection point of two lines.
    """
    a0, b0, c0 = line0
    a1, b1, c1 = line1
    D = a0 * b1 - a1 * b0  # D==0 then two lines overlap
    D[D == 0] = np.nan
    x = (b0 * c1 - b1 * c0) / D  # x and y can be nan if D==0, which will be handled in the later steps
    y = (a1 * c0 - a0 * c1) / D
    return np.array([x, y])


def ison(line_start, line_end, point, tol=1e-5):
    '''
    Check if a point is on a line segment.
    tol is the tolerance for considering the point is on the line segment.
    '''
    crossproduct = (point[1] - line_start[1]) * (line_end[0] - line_start[0]) - (point[0] - line_start[0]) * (
                line_end[1] - line_start[1])
    dotproduct = (point[0] - line_start[0]) * (line_end[0] - line_start[0]) + (point[1] - line_start[1]) * (
                line_end[1] - line_start[1])
    squaredlength = (line_end[0] - line_start[0]) ** 2 + (line_end[1] - line_start[1]) ** 2
    return (np.absolute(crossproduct) <= tol) & (dotproduct >= 0) & (dotproduct <= squaredlength)


def dist_p2l(point, line_start, line_end):
    '''
    Get the distance from a point to a line.
    '''
    return np.absolute((line_end[0] - line_start[0]) * (line_start[1] - point[1]) - (line_start[0] - point[0]) * (
                line_end[1] - line_start[1])) / np.sqrt(
        (line_end[0] - line_start[0]) ** 2 + (line_end[1] - line_start[1]) ** 2)


def getpoints(samples):
    '''
    Get the four points of the bounding box of vehicles i and j.
    '''
    # vehicle i
    heading_i = samples[['hx_i', 'hy_i']].values
    perp_heading_i = np.array([-heading_i[:, 1], heading_i[:, 0]]).T
    heading_scale_i = np.tile(np.sqrt(heading_i[:, 0] ** 2 + heading_i[:, 1] ** 2), (2, 1)).T
    length_i = np.tile(samples.length_i.values, (2, 1)).T
    width_i = np.tile(samples.width_i.values, (2, 1)).T

    point_up = samples[['x_i', 'y_i']].values + heading_i / heading_scale_i * length_i / 2
    point_down = samples[['x_i', 'y_i']].values - heading_i / heading_scale_i * length_i / 2
    point_i1 = (point_up + perp_heading_i / heading_scale_i * width_i / 2).T
    point_i2 = (point_up - perp_heading_i / heading_scale_i * width_i / 2).T
    point_i3 = (point_down + perp_heading_i / heading_scale_i * width_i / 2).T
    point_i4 = (point_down - perp_heading_i / heading_scale_i * width_i / 2).T

    # vehicle j
    heading_j = samples[['hx_j', 'hy_j']].values
    perp_heading_j = np.array([-heading_j[:, 1], heading_j[:, 0]]).T
    heading_scale_j = np.tile(np.sqrt(heading_j[:, 0] ** 2 + heading_j[:, 1] ** 2), (2, 1)).T
    length_j = np.tile(samples.length_j.values, (2, 1)).T
    width_j = np.tile(samples.width_j.values, (2, 1)).T

    point_up = samples[['x_j', 'y_j']].values + heading_j / heading_scale_j * length_j / 2
    point_down = samples[['x_j', 'y_j']].values - heading_j / heading_scale_j * length_j / 2
    point_j1 = (point_up + perp_heading_j / heading_scale_j * width_j / 2).T
    point_j2 = (point_up - perp_heading_j / heading_scale_j * width_j / 2).T
    point_j3 = (point_down + perp_heading_j / heading_scale_j * width_j / 2).T
    point_j4 = (point_down - perp_heading_j / heading_scale_j * width_j / 2).T

    return (point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4)


def CurrentD(samples, toreturn='dataframe'):
    '''
    Compute the distance between the bounding boxes of vehicles i and j (0 if overlapping).
    '''
    if toreturn != 'dataframe' and toreturn != 'values':
        warnings.warn("Incorrect target to return. Please specify \'dataframe\' or \'values\'.")
    else:
        point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4 = getpoints(samples)

        dist_mat = []
        count_i = 0
        for point_i_start, point_i_end in zip([point_i1, point_i4, point_i3, point_i2],
                                              [point_i2, point_i3, point_i1, point_i4]):
            count_j = 0
            for point_j_start, point_j_end in zip([point_j1, point_j4, point_j3, point_j2],
                                                  [point_j2, point_j3, point_j1, point_j4]):
                if count_i < 2 and count_j < 2:
                    # Distance from point to point
                    dist_mat.append(np.sqrt(
                        (point_i_start[0] - point_j_start[0]) ** 2 + (point_i_start[1] - point_j_start[1]) ** 2))
                    dist_mat.append(
                        np.sqrt((point_i_start[0] - point_j_end[0]) ** 2 + (point_i_start[1] - point_j_end[1]) ** 2))
                    dist_mat.append(
                        np.sqrt((point_i_end[0] - point_j_start[0]) ** 2 + (point_i_end[1] - point_j_start[1]) ** 2))
                    dist_mat.append(
                        np.sqrt((point_i_end[0] - point_j_end[0]) ** 2 + (point_i_end[1] - point_j_end[1]) ** 2))

                # Distance from point to edge
                ist = intersect(line(point_i_start, point_i_start + np.array(
                    [-(point_j_start - point_j_end)[1], (point_j_start - point_j_end)[0]])),
                                line(point_j_start, point_j_end))
                ist[:, ~ison(point_j_start, point_j_end, ist, tol=1e-2)] = np.nan
                dist_mat.append(np.sqrt((ist[0] - point_i_start[0]) ** 2 + (ist[1] - point_i_start[1]) ** 2))

                # Overlapped bounding boxes
                ist = intersect(line(point_i_start, point_i_end), line(point_j_start, point_j_end))
                dist = np.ones(len(samples)) * np.nan
                dist[ison(point_i_start, point_i_end, ist) & ison(point_j_start, point_j_end, ist)] = 0
                dist[np.isnan(ist[0]) & (
                            ison(point_i_start, point_i_end, point_j_start) | ison(point_i_start, point_i_end,
                                                                                   point_j_end))] = 0
                dist_mat.append(dist)
                count_j += 1
            count_i += 1

        cdist = np.nanmin(np.array(dist_mat), axis=0)

        if toreturn == 'dataframe':
            samples['CurrentD'] = cdist
            return samples
        elif toreturn == 'values':
            return cdist


def DTC_ij(samples):
    ''''''
    point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4 = getpoints(samples)
    relative_v = (samples[['vx_i', 'vy_i']].values - samples[['vx_j', 'vy_j']].values).T

    dist_mat = []
    leaving_mat = []
    # For each point of vehicle i
    for point_line_start in [point_i1, point_i2, point_i3, point_i4]:
        # For each edge of vehicle j
        for edge_start, edge_end in zip([point_j1, point_j3, point_j1, point_j2],
                                        [point_j2, point_j4, point_j3, point_j4]):
            point_line_end = point_line_start + relative_v
            # intersection point between 
            # 1) the edge of vehicle j and 
            # 2) the line extended from the point of vehicle i along the direction of the relative velocity of vehicle i and j
            ist = intersect(line(point_line_start, point_line_end), line(edge_start, edge_end))
            ist[:, ~ison(edge_start, edge_end, ist, tol=1e-2)] = np.nan
            # distance from the point of vehicle i to the intersection point
            dist_ist = np.sqrt((ist[0] - point_line_start[0]) ** 2 + (ist[1] - point_line_start[1]) ** 2)
            dist_ist[np.isnan(dist_ist)] = np.inf
            dist_mat.append(dist_ist)
            # determine if vehicle i and vehicle j are leaving each other based on if
            # 1) the relative velocity of vehicle i and j and
            # 2) the vector from the point of vehicle i to the intersection point
            # are in the same direction (>=0) or the opposite direction (<0)
            leaving = relative_v[0] * (ist[0] - point_line_start[0]) + relative_v[1] * (ist[1] - point_line_start[1])
            leaving[leaving >= 0] = 20
            leaving[leaving < 0] = 1
            leaving_mat.append(leaving)

    dtc = np.array(dist_mat).min(axis=0)
    leaving = np.nansum(np.array(leaving_mat), axis=0)
    return dtc, leaving


# SSM computation
def TTC(samples, toreturn='dataframe'):
    if toreturn != 'dataframe' and toreturn != 'values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    else:
        delta_v = np.sqrt((samples['vx_i'] - samples['vx_j']) ** 2 + (samples['vy_i'] - samples['vy_j']) ** 2)
        dtc_ij, leaving_ij = DTC_ij(samples)
        ttc_ij = dtc_ij / delta_v
        ttc_ij[leaving_ij < 20] = np.inf  # inf means the two vehicles will not collide if they keep current velocity
        ttc_ij[(leaving_ij > 20) & (
                    leaving_ij % 20 != 0)] = -1  # -1 means the bounding boxes of the two vehicles are overlapping

        keys = [var + '_i' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
        values = [var + '_j' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
        keys.extend(values)
        values.extend(keys)
        rename_dict = {keys[i]: values[i] for i in range(len(keys))}
        dtc_ji, leaving_ji = DTC_ij(samples.rename(columns=rename_dict))
        ttc_ji = dtc_ji / delta_v
        ttc_ji[leaving_ji < 20] = np.inf
        ttc_ji[(leaving_ji > 20) & (leaving_ji % 20 != 0)] = -1

        if toreturn == 'dataframe':
            samples['TTC'] = np.minimum(ttc_ij, ttc_ji)
            return samples
        elif toreturn == 'values':
            return np.minimum(ttc_ij, ttc_ji)


def DRAC(samples, toreturn='dataframe'):
    if toreturn != 'dataframe' and toreturn != 'values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    else:
        delta_v = np.sqrt((samples['vx_i'] - samples['vx_j']) ** 2 + (samples['vy_i'] - samples['vy_j']) ** 2)
        dtc_ij, leaving_ij = DTC_ij(samples)
        drac_ij = delta_v ** 2 / dtc_ij / 2
        drac_ij[leaving_ij < 20] = 0.  # the two vehicles will not collide if they keep current velocity
        drac_ij[(leaving_ij > 20) & (
                    leaving_ij % 20 != 0)] = -1  # -1 means the bounding boxes of the two vehicles are overlapping

        keys = [var + '_i' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
        values = [var + '_j' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
        keys.extend(values)
        values.extend(keys)
        rename_dict = {keys[i]: values[i] for i in range(len(keys))}
        dtc_ji, leaving_ji = DTC_ij(samples.rename(columns=rename_dict))
        drac_ji = delta_v ** 2 / dtc_ji / 2
        drac_ji[leaving_ji < 20] = 0.
        drac_ji[(leaving_ji > 20) & (leaving_ji % 20 != 0)] = -1

        if toreturn == 'dataframe':
            samples['DRAC'] = np.maximum(drac_ij, drac_ji)
            return samples
        elif toreturn == 'values':
            return np.maximum(drac_ij, drac_ji)


def MTTC(samples, toreturn='dataframe'):
    if toreturn != 'dataframe' and toreturn != 'values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    elif 'acc_i' not in samples.columns:
        warnings.warn('Acceleration of the ego vehicle is not provided.')
    else:
        delta_v = np.sqrt((samples['vx_i'] - samples['vx_j']) ** 2 + (samples['vy_i'] - samples['vy_j']) ** 2)
        dtc_ij, leaving_ij = DTC_ij(samples)
        ttc_ij = dtc_ij / delta_v
        ttc_ij[leaving_ij < 20] = np.inf  # inf means the two vehicles will not collide if they keep current velocity
        ttc_ij[(leaving_ij > 20) & (
                    leaving_ij % 20 != 0)] = -1  # -1 means the bounding boxes of the two vehicles are overlapping

        keys = [var + '_i' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
        values = [var + '_j' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
        keys.extend(values)
        values.extend(keys)
        rename_dict = {keys[i]: values[i] for i in range(len(keys))}
        dtc_ji, leaving_ji = DTC_ij(samples.rename(columns=rename_dict))
        ttc_ji = dtc_ji / delta_v
        ttc_ji[leaving_ji < 20] = np.inf
        ttc_ji[(leaving_ji > 20) & (leaving_ji % 20 != 0)] = -1

        ttc = np.minimum(ttc_ij, ttc_ji)
        dtc = np.minimum(dtc_ij, dtc_ji)

        if 'acc_j' in samples.columns:
            acc_i = samples['acc_i'].values
            acc_j = samples['acc_j'].values
            delta_a = acc_i - acc_j
        else:  # assume acc_j=0 (i.e., the other vehicle keeps current velocity)
            acc_i = samples['acc_i'].values
            delta_a = acc_i
        delta_v = delta_v * np.sign(((leaving_ij >= 20) | (leaving_ji >= 20)).astype(
            int) - 0.5)  # if the two vehicles are leaving each other, the relative velocity is set negative
        squared_term = delta_v ** 2 + 2 * delta_a * dtc
        squared_term[squared_term >= 0] = np.sqrt(squared_term[squared_term >= 0])
        squared_term[squared_term < 0] = np.nan
        mttc_plus = (-delta_v + squared_term) / delta_a
        mttc_minus = (-delta_v - squared_term) / delta_a
        mttc = mttc_minus.copy()
        mttc[(mttc_minus <= 0) & (mttc_plus > 0)] = mttc_plus[(mttc_minus <= 0) & (mttc_plus > 0)]
        mttc[(mttc_minus <= 0) & (mttc_plus <= 0)] = np.inf
        mttc[(np.isnan(mttc_minus) | np.isnan(mttc_plus))] = np.inf
        mttc[abs(delta_a) < 1e-6] = ttc[abs(delta_a) < 1e-6]
        mttc[((leaving_ij > 20) & (leaving_ij % 20 != 0)) | ((leaving_ji > 20) & (leaving_ji % 20 != 0))] = -1

        if toreturn == 'dataframe':
            samples['MTTC'] = mttc
            return samples
        elif toreturn == 'values':
            return mttc


def TTC_DRAC_MTTC(samples, toreturn='dataframe'):
    if toreturn != 'dataframe' and toreturn != 'values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    elif 'acc_i' not in samples.columns:
        warnings.warn('Acceleration of the ego vehicle is not provided.')
    else:
        delta_v = np.sqrt((samples['vx_i'] - samples['vx_j']) ** 2 + (samples['vy_i'] - samples['vy_j']) ** 2)
        dtc_ij, leaving_ij = DTC_ij(samples)
        ttc_ij = dtc_ij / delta_v
        ttc_ij[leaving_ij < 20] = np.inf  # inf means the two vehicles will not collide if they keep current velocity
        ttc_ij[(leaving_ij > 20) & (
                    leaving_ij % 20 != 0)] = -1  # -1 means the bounding boxes of the two vehicles are overlapping
        drac_ij = delta_v ** 2 / dtc_ij / 2
        drac_ij[leaving_ij < 20] = 0.  # the two vehicles will not collide if they keep current velocity
        drac_ij[(leaving_ij > 20) & (
                    leaving_ij % 20 != 0)] = -1  # -1 means the bounding boxes of the two vehicles are overlapping

        keys = [var + '_i' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
        values = [var + '_j' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
        keys.extend(values)
        values.extend(keys)
        rename_dict = {keys[i]: values[i] for i in range(len(keys))}
        dtc_ji, leaving_ji = DTC_ij(samples.rename(columns=rename_dict))
        ttc_ji = dtc_ji / delta_v
        ttc_ji[leaving_ji < 20] = np.inf
        ttc_ji[(leaving_ji > 20) & (leaving_ji % 20 != 0)] = -1
        drac_ji = delta_v ** 2 / dtc_ji / 2
        drac_ji[leaving_ji < 20] = 0.
        drac_ji[(leaving_ji > 20) & (leaving_ji % 20 != 0)] = -1

        dtc = np.minimum(dtc_ij, dtc_ji)
        ttc = np.minimum(ttc_ij, ttc_ji)
        drac = np.maximum(drac_ij, drac_ji)

        if 'acc_j' in samples.columns:
            acc_i = samples['acc_i'].values
            acc_j = samples['acc_j'].values
            delta_a = acc_i - acc_j
        else:  # assume acc_j=0 (i.e., the other vehicle keeps current velocity)
            acc_i = samples['acc_i'].values
            delta_a = acc_i
        delta_v = delta_v * np.sign(((leaving_ij >= 20) | (leaving_ji >= 20)).astype(
            int) - 0.5)  # if the two vehicles are leaving each other, the relative velocity is set negative
        squared_term = delta_v ** 2 + 2 * delta_a * dtc
        squared_term[squared_term >= 0] = np.sqrt(squared_term[squared_term >= 0])
        squared_term[squared_term < 0] = np.nan
        mttc_plus = (-delta_v + squared_term) / delta_a
        mttc_minus = (-delta_v - squared_term) / delta_a
        mttc = mttc_minus.copy()
        mttc[(mttc_minus <= 0) & (mttc_plus > 0)] = mttc_plus[(mttc_minus <= 0) & (mttc_plus > 0)]
        mttc[(mttc_minus <= 0) & (mttc_plus <= 0)] = np.inf
        mttc[(np.isnan(mttc_minus) | np.isnan(mttc_plus))] = np.inf
        mttc[abs(delta_a) < 1e-6] = ttc[abs(delta_a) < 1e-6]
        mttc[((leaving_ij > 20) & (leaving_ij % 20 != 0)) | ((leaving_ji > 20) & (leaving_ji % 20 != 0))] = -1

        if toreturn == 'dataframe':
            samples['TTC'] = ttc
            samples['DRAC'] = drac
            samples['MTTC'] = mttc
            return samples
        elif toreturn == 'values':
            return ttc, drac, mttc


# Efficiency evaluation
def efficiency(samples, indicator, iterations):
    if indicator == 'TTC':
        compute_func = TTC
    elif indicator == 'DRAC':
        compute_func = DRAC
    elif indicator == 'MTTC':
        compute_func = MTTC
    elif indicator == 'TTC_DRAC_MTTC':
        compute_func = TTC_DRAC_MTTC
    else:
        print('Incorrect indicator. Please specify \'TTC\', \'DRAC\', \'MTTC\', or \'TTC_DRAC_MTTC\'.')
        return None
    import time as systime
    ts = []
    for _ in range(iterations):
        t = systime.time()
        _ = compute_func(samples, 'values')
        ts.append(systime.time() - t)
    return sum(ts) / iterations


def traj_data_TTC(traj_i: pd.DataFrame, traj_j: pd.DataFrame, is_ori=False):
    # 转化为标准数据Dataframe
    x_i = traj_i[TI.xCenterGlobal] if not is_ori else traj_i[TI.xCenterGlobal + '_ori']
    y_i = traj_i[TI.yCenterGlobal] if not is_ori else traj_i[TI.yCenterGlobal + '_ori']
    vx_i = traj_i[TI.speed] * np.cos(traj_i[TI.yaw]) if not is_ori else traj_i[TI.speed + "_ori"] * np.cos(
        traj_i[TI.yaw + '_ori'])
    vy_i = traj_i[TI.speed] * np.sin(traj_i[TI.yaw]) if not is_ori else traj_i[TI.speed + "_ori"] * np.sin(
        traj_i[TI.yaw + '_ori'])
    hx_i = np.cos(traj_i[TI.yaw]) if not is_ori else np.cos(traj_i[TI.yaw + '_ori'])
    hy_i = np.sin(traj_i[TI.yaw]) if not is_ori else np.sin(traj_i[TI.yaw + '_ori'])
    length_i = traj_i[TI.length]
    width_i = traj_i[TI.width]  # ATTENTION

    x_j = traj_j[TI.xCenterGlobal] if not is_ori else traj_j[TI.xCenterGlobal + '_ori']
    y_j = traj_j[TI.yCenterGlobal] if not is_ori else traj_j[TI.yCenterGlobal + '_ori']
    vx_j = traj_j[TI.speed] * np.cos(traj_j[TI.yaw]) if not is_ori else traj_j[TI.speed + "_ori"] * np.cos(
        traj_j[TI.yaw + '_ori'])
    vy_j = traj_j[TI.speed] * np.sin(traj_j[TI.yaw]) if not is_ori else traj_j[TI.speed + "_ori"] * np.sin(
        traj_j[TI.yaw + '_ori'])
    hx_j = np.cos(traj_j[TI.yaw]) if not is_ori else np.cos(traj_j[TI.yaw + '_ori'])
    hy_j = np.sin(traj_j[TI.yaw]) if not is_ori else np.sin(traj_j[TI.yaw + '_ori'])
    length_j = traj_j[TI.length]
    width_j = traj_j[TI.width]  # ATTENTION

    df = pd.DataFrame({
        'x_i': x_i,
        'y_i': y_i,
        'vx_i': vx_i,
        'vy_i': vy_i,
        'hx_i': hx_i,
        'hy_i': hy_i,
        'length_i': length_i,
        'width_i': width_i,
        'x_j': x_j,
        'y_j': y_j,
        'vx_j': vx_j,
        'vy_j': vy_j,
        'hx_j': hx_j,
        'hy_j': hy_j,
        'length_j': length_j,
        'width_j': width_j
    })

    res = TTC(df, toreturn='values')
    return res


def calculate_collision_risk(traj_i: list[TrajPoint], traj_j: list[TrajPoint]):
    # -----------------------------------------------------------------------------------
    # x_i      :  x coordinate of the ego vehicle (usually assumed to be centroid)      |
    # y_i      :  y coordinate of the ego vehicle (usually assumed to be centroid)      |
    # vx_i     :  x coordinate of the velocity of the ego vehicle                       |
    # vy_i     :  y coordinate of the velocity of the ego vehicle                       |
    # hx_i     :  x coordinate of the heading direction of the ego vehicle              |
    # hy_i     :  y coordinate of the heading direction of the ego vehicle              |
    # length_i :  length of the ego vehicle                                             |
    # width_i  :  width of the ego vehicle                                              |
    # x_j      :  x coordinate of another vehicle (usually assumed to be centroid)      |
    # y_j      :  y coordinate of another vehicle (usually assumed to be centroid)      |
    # vx_j     :  x coordinate of the velocity of another vehicle                       |
    # vy_j     :  y coordinate of the velocity of another vehicle                       |
    # hx_j     :  x coordinate of the heading direction of another vehicle              |
    # hy_j     :  y coordinate of the heading direction of another vehicle              |
    # length_j :  length of another vehicle                                             |
    # width_j  :  width of another vehicle                                              |
    #------------------------------------------------------------------------------------
    # 转化为标准数据Dataframe
    traj_i = np.array([point.to_center() for point in traj_i])
    x_i = traj_i[:, 0]
    y_i = traj_i[:, 3]
    vx_i = traj_i[:, 1]
    vy_i = traj_i[:, 4]
    hx_i = traj_i[:, 6]
    hy_i = traj_i[:, 7]
    length_i = traj_i[:, 8]
    width_i = traj_i[:, 9] * 0.1  # ATTENTION

    traj_j = np.array([point.to_center() for point in traj_j])
    x_j = traj_j[:, 0]
    y_j = traj_j[:, 3]
    vx_j = traj_j[:, 1]
    vy_j = traj_j[:, 4]
    hx_j = traj_j[:, 6]
    hy_j = traj_j[:, 7]
    length_j = traj_j[:, 8]
    width_j = traj_j[:, 9] * 0.1  # ATTENTION
    df = pd.DataFrame({
        'x_i': x_i,
        'y_i': y_i,
        'vx_i': vx_i,
        'vy_i': vy_i,
        'hx_i': hx_i,
        'hy_i': hy_i,
        'length_i': length_i,
        'width_i': width_i,
        'x_j': x_j,
        'y_j': y_j,
        'vx_j': vx_j,
        'vy_j': vy_j,
        'hx_j': hx_j,
        'hy_j': hy_j,
        'length_j': length_j,
        'width_j': width_j
    })
    ttc_2d = TTC(df, toreturn='values')
    ttc_2d[ttc_2d < 0] = -np.inf
    return ttc_2d


# 计算目标车辆角点与前车或后车的碰撞概率
def compute_collision_probability(x_min, x_max, y_min, y_max, param):
    """
    计算目标车辆角点与前车或后车的碰撞概率
    :param x_min: 目标车辆角点的最小x坐标
    :param x_max: 目标车辆角点的最大x坐标
    :param y_min: 目标车辆角点的最小y坐标
    :param y_max: 目标车辆角点的最大y坐标
    :param center_point: 目标车辆中心点坐标
    :param param: 前车或后车的轨迹分布参数 (sigma_x, sigma_y, rho)
    """
    mu_x, mu_y, sigma_x, sigma_y, rho = param
    cov = np.array([[sigma_x ** 2, rho * sigma_x * sigma_y],
                    [rho * sigma_x * sigma_y, sigma_y ** 2]])
    multivariate_normal_pdf = multivariate_normal(mean=(mu_x, mu_y), cov=cov)

    # 蒙特卡洛方法估算概率
    num_samples = 1000
    samples = multivariate_normal_pdf.rvs(size=num_samples)
    mask = (samples[:, 0] >= x_min) & (samples[:, 0] <= x_max) & \
           (samples[:, 1] >= y_min) & (samples[:, 1] <= y_max)
    probability_mc = np.sum(mask) / num_samples

    return probability_mc


# 计算最大化的碰撞概率
def calculate_max_collision_risk_by_prob(
        target_traj, target_width, target_length,
        front_traj_prob, front_width, front_length,
        rear_traj_prob, rear_width, rear_length,
        ori_front_traj, ori_front_width, ori_front_length,
):
    """
    计算目标车辆与前后车在每个时间步的角点碰撞概率，并找到最大碰撞概率
    target_traj_bbox: 目标车辆的角点轨迹 (N, 4, 2)
    front_traj: 前车的角点轨迹 (N, 2)
    param_front: 前车的轨迹分布参数 (mu_x, mu_y, sigma_x, sigma_y, rho)
    rear_traj: 后车的角点轨迹 (N, 2)
    param_rear: 后车的轨迹分布参数 (mu_x, mu_y, sigma_x, sigma_y, rho)
    """
    collision_p_list = []
    # 对每个时间步计算目标车与前后车的角点碰撞概率
    for t in range(len(target_traj)):
        target_point = target_traj[t]

        # 计算四个角点与前车和后车的碰撞概率
        x_min, x_max = target_point[0] - target_length / 2, target_point[0] + target_length / 2
        y_min, y_max = target_point[1] - target_width / 2, target_point[1] + target_width / 2

        # x_min_f = x_min - front_length / 2
        # x_max_f = x_max + front_length / 2
        # y_min_f = y_min - front_width / 2
        # y_max_f = y_max + front_width / 2
        x_min_f = x_min
        x_max_f = x_max
        y_min_f = y_min
        y_max_f = y_max
        front_point = front_traj_prob[t]
        front_prob = compute_collision_probability(x_min_f, x_max_f, y_min_f, y_max_f, front_point)

        # 计算与后车的碰撞概率
        # x_min_r = x_min - rear_length / 2
        # x_max_r = x_max + rear_length / 2
        # y_min_r = y_min - rear_width / 2
        # y_max_r = y_max + rear_width / 2
        x_min_r = x_min
        x_max_r = x_max
        y_min_r = y_min
        y_max_r = y_max
        rear_point = rear_traj_prob[t]
        rear_prob = compute_collision_probability(x_min_r, x_max_r, y_min_r, y_max_r, rear_point)

        # 计算与前车的碰撞概率
        # x_min_ori = x_min - ori_front_length / 2
        # x_max_ori = x_max + ori_front_length / 2
        # y_min_ori = y_min - ori_front_width / 2
        # y_max_ori = y_max + ori_front_width / 2
        x_min_ori = x_min
        x_max_ori = x_max
        y_min_ori = y_min
        y_max_ori = y_max
        front_ori_point = ori_front_traj[t]
        front_ori_prob = compute_collision_probability(x_min_ori, x_max_ori, y_min_ori, y_max_ori, front_ori_point)

        # 选择当前角点的最大碰撞概率
        max_prob = max(front_prob, rear_prob, front_ori_prob)
        collision_p_list.append(max_prob)

    # 计算所有时间步的最大碰撞概率
    max_collision_probability = max(collision_p_list)
    return max_collision_probability
