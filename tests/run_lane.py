# -*- coding = uft-8 -*-
# @Time : 2023-04-09 20:20
# @Author : yzbyx
# @File : run_open.py
# @Software : PyCharm

from trasim_simplified.core.constant import CFM, V_TYPE, COLOR
from trasim_simplified.core.frame.micro.circle_lane import LaneCircle
from trasim_simplified.core.frame.micro.open_lane import LaneOpen
from trasim_simplified.util.timer import timer_no_log


@timer_no_log
def run_circle():
    _cf_param = {"original_acc": False, "v_safe_dispersed": True}
    _car_param = {}
    take_over_index = -1
    follower_index = [-1]
    dt = 0.1
    warm_up_step = int(1800 / dt)
    sim_step = warm_up_step + int(3600 / dt)
    offset_step = int(500 / dt) + warm_up_step
    dec_step = int(50 / dt) + offset_step
    maintain_step = int(100 / dt) + dec_step
    acc_step = int(50 / dt) + maintain_step

    is_circle = False
    lane_length = 10000

    car_id_list = None

    if is_circle:
        sim = LaneCircle(lane_length)
        sim.car_config(
            0.1,
            7.5,
            V_TYPE.PASSENGER,
            0,
            False,
            CFM.IDM,
            _cf_param,
            {"color": COLOR.yellow},
        )
        # sim.car_config(50, 7.5, V_TYPE.PASSENGER, 20, False, CFM.NON_LINEAR_GHR, _cf_param, {"color": COLOR.blue})
        sim.car_load(5)
    else:
        sim = LaneOpen(lane_length)
        sim.car_config(
            50, 7.5, V_TYPE.PASSENGER, 0, False, CFM.IDM, _cf_param, _car_param
        )
        sim.car_loader(2000)

    sim.data_container.config()
    for step in sim.run(
        data_save=True,
        has_ui=False,
        frame_rate=-1,
        warm_up_step=warm_up_step,
        sim_step=sim_step,
        dt=dt,
    ):
        # 头车控制车队扰动
        # if step < 30 / dt:
        #     sim.take_over(car_id_list[-1], 1)
        # else:
        #     sim.take_over(car_id_list[-1], 0)
        # if warm_up_step < step <= offset_step:
        #     sim.take_over(car_id_list[-1], 0)

        # 车辆减速扰动
        if step == offset_step:
            take_over_index = sim.get_appropriate_car()
            follower_index = [
                sim.get_relative_id(take_over_index, -i - 1) for i in range(5)
            ]
            print(take_over_index, follower_index)
        if offset_step < step <= dec_step:
            sim.take_over(take_over_index, -1)
        if dec_step < step <= maintain_step:
            sim.take_over(take_over_index, 0)
        if maintain_step < step <= acc_step:
            sim.take_over(take_over_index, 1)
        if step > acc_step:
            sim.take_over(take_over_index, 0)

        # 居中插入车辆
        # if offset_step == step:
        #     take_over_index = sim.get_appropriate_car()
        #     take_over_speed = sim.get_car_info(take_over_index, C_Info.v)
        #     follower_index = sim.car_insert_middle(
        #         7.5, V_TYPE.PASSENGER, take_over_speed, 0, CFM.IDM,
        #         _cf_param, {"color": COLOR.yellow}, take_over_index
        #     )
        #     print(take_over_index, follower_index)

        pass

    df = sim.data_container.data_to_df()
    df.to_csv(r"D:\\test.csv", index=False)
    # sim.data_container.save(
    #     df, r"E:\PyProject\car-following-model-test\tests\thesis_experiment\data\TPACC_OCC0.25"
    # )
    # result = sim.data_processor.aggregate_as_detect_loop(df, lane_id=0, lane_length=lane_length, pos=0, width=900,
    #                                                      dt=dt, d_step=int(300 / dt))
    # print(sim.data_processor.circle_kqv_cal(df, lane_length))
    # sim.data_processor.print(result)

    # print(f"TET_sum: {np.sum(sim.data_container.data_df[C_Info.safe_tet])}")
    # print(f"TIT_sum: {np.sum(sim.data_container.data_df[C_Info.safe_tit])}")

    # axes = Plot.basic_plot(
    #     follower_index, lane_id=-1, data_df=df, time_range=(offset_step, acc_step + 50)
    # )
    # Plot.add_plot_2D(axes[0, 1], func_=lambda x: x * 1.4, x_step=1)
    # Plot.add_plot_2D(axes[0, 1], func_=lambda x: x, x_step=1)
    # Plot.spatial_time_plot(follower_index[0], lane_id=0, color_info_name=C_Info.safe_picud_KK, data_df=df,
    #                        color_lambda_=lambda x: -1 if x < 0 else 0)
    # Plot.spatial_time_plot(follower_index[0], lane_add_num=0, data_df=df)
    # # Plot.spatial_time_plot(follower_index[0], lane_id=0, color_info_name=C_Info.a, data_df=df,
    # #                        color_lambda_=lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    # Plot.spatial_time_plot(follower_index[0], lane_id=0, color_info_name=C_Info.v, data_df=df)

    # Plot.plot_density_map(
    #     df, 0, dt, int(300 / dt), 50, [warm_up_step, sim_step], [0, lane_length]
    # )
    #
    # Plot.show()

    # sim.data_processor.aggregate_as_detect_loop(0, 995, 6000)
    # sim.data_processor.print_result()


if __name__ == "__main__":
    run_circle()
