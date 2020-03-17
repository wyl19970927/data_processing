import pandas as pd
import numpy as np
import math
import time
import datetime

project_path = {
                "Pj4419010003": './data/_虎门数据整理/',     # 东莞虎门
                "Pj3206120001": './data/_南通通州数据整理/',     # 南通通州
                "Pj4413030001": './data/_大亚湾数据整理/',     # 惠州大亚湾
                "Pj5101220001": './data/_双流数据整理/',     # 成都双流
                "Pj5101050001": './data/_青羊数据整理/',     # 成都青羊
                "Pj3205830001": './data/_昆山数据整理/',     # 苏州昆山
}
dicts = {
    'ch_energy': ['制冷机房主机用电', '冷冻站冷冻机组', '制冷机', '冷水机组'],
    'ev_pump_energy': ['冷冻泵', '冷冻水泵', '冷冻水循环泵'],
    'cd_pump_energy': ['冷却泵', '冷却水泵', '冷却水循环泵'],
    'ct_energy': ['冷却塔']
}
project_name = {
                "Pj4419010003": 'HumenWanDa',     # 东莞虎门
                "Pj3206120001": 'TongzhouWanDa',     # 南通通州
                "Pj4413030001": 'DayawanWanDa',   # 惠州大亚湾
                "Pj5101220001": 'ShuangLiuWanDa',     # 成都双流
                "Pj5101050001": 'QingYangWanDa',     # 成都青羊
                "Pj3205830001": 'KunshanWanDa',     # 苏州昆山
}
def time_convert(time_str):
    return datetime.datetime.strptime(time_str,'%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H')
def time_convert_energy(time_str):
    return datetime.datetime.strptime(time_str,'%Y/%m/%d %H').strftime('%Y-%m-%d %H')
def time_convert_weather(time_str):
    return datetime.datetime.strptime(time_str,'%Y/%m/%d %H:%M:%S').strftime('%Y-%m-%d %H')
def processing_time(CCS_filepath):
    df_CCS = pd.read_excel(CCS_filepath)
    np_time = np.array(df_CCS['Time'])
    indes = np.where(np.array([np_time[i][-5:-3] for i in range(len(np_time))]) == '00')
    Times = np_time[indes]
    return  Times
#sheet==Indoor
def data_indoor(filepath,CCS_filepath):

    df_s = pd.read_excel(filepath)


    new_df = df_s.pivot(index='COUNTTIME', columns='SIGN', values='DATA')
    new_df = new_df.reset_index()

    np_time = processing_time(CCS_filepath)

    np_time2 = []
    for np_times in np_time:
        np_time2.append(time_convert(np_times))
    TimeS = np.array(new_df['COUNTTIME'])
    TimeS2=[]
    for TimeSs in TimeS :
       TimeS2.append(time_convert(TimeSs))
    # TimeS=time_convert(TimeS)
    inde_s = []
    ins = []
    sametime = [x for x in np_time2 if x in TimeS2]

    indesame=[]

    for i in sametime:
        indesame.append(TimeS2.index(i))


    for i, j in enumerate(TimeS):
        ins.append(i)


    w = [x for x in ins if x not in indesame]
    for s in  w:
        new_df = new_df.drop(s, axis=0, inplace=False)

    df = new_df
    columns_all = df.columns.values

    column_s1=[]
    column_s2 = []
    for i in range(1, df.shape[1]):
        num = 0
        for j in range(0, df.shape[0] - 1):
            if df.iloc[j, i] == df.iloc[j + 1, i]:
                num += 1
                if (num == 20):
                    column_s1.append(columns_all[i])
                    break
            else:
                num = 0

    for i in range(1, df.shape[1]):
        for j in range(0, df.shape[0] - 1):
            if pd.isnull(df.iloc[j, i]) == True:
                column_s2.append(columns_all[i])
                break
    for p in column_s1:
        df = df.drop(p, axis=1, inplace=False)

    a = [x for x in column_s2 if x not in column_s1]
    # b = [y for y in (column_s1 + column_s2) if y not in a]

    for s in a:
        df = df.drop(s, axis=1, inplace=False)


    return df
#sheet==energy
def data_energy(filepath,CCS_filepath):

    df = pd.read_excel(filepath)

    np_time = processing_time(CCS_filepath)

    TimeS = np.array(df['time'])



    np_time2 = []
    TimeS2 = []
    for np_times in np_time:
        np_time2.append(time_convert(np_times))
    for j in TimeS:
        TimeS2.append(time_convert_energy(j))
    sametime = [x for x in np_time2 if x in TimeS2]
    indesame = []
    for i in sametime:
        indesame.append(TimeS2.index(i))


    all_eq_dict = {'ch_energy': [], 'ev_pump_energy': [], 'cd_pump_energy': [], 'ct_energy': []}

    columns_all = df.columns.values
    ch_energy = []
    ev_pump_energy = []
    cd_pump_energy = []
    ct_energy = []

    ch_energy_sum = []
    ev_pump_energy_sum = []
    cd_pump_energy_sum = []
    ct_energy_sum = []
    plant_energy = []

    for eq_name in columns_all:
        for eq_code, eq_list in dicts.items():
            for i in eq_list:
                if i in eq_name and eq_name not in all_eq_dict[eq_code]:
                    all_eq_dict[eq_code].append(eq_name)

    for key in all_eq_dict:
        for i in range(len(all_eq_dict[key])):
            if key == 'ch_energy':
                ch_energy.append(list(df[str(all_eq_dict[key][i])]))
            elif key == 'ev_pump_energy':
                ev_pump_energy.append(list(df[str(all_eq_dict[key][i])]))
            elif key == 'cd_pump_energy':
                cd_pump_energy.append(list(df[str(all_eq_dict[key][i])]))
            elif key == 'ct_energy':
                ct_energy.append(list(df[str(all_eq_dict[key][i])]))

    for i in range(len(ch_energy[0])):
        x = 0
        for j in range(len(ch_energy)):
            x += ch_energy[j][i]
        ch_energy_sum.append(x)
    for i in range(len(ev_pump_energy[0])):
        x = 0
        for j in range(len(ev_pump_energy)):
            x += ev_pump_energy[j][i]
        ev_pump_energy_sum.append(x)
    for i in range(len(cd_pump_energy[0])):
        x = 0
        for j in range(len(cd_pump_energy)):
            x += cd_pump_energy[j][i]
        cd_pump_energy_sum.append(x)
    for i in range(len(ct_energy[0])):
        x = 0
        for j in range(len(ct_energy)):
            x += ct_energy[j][i]
        ct_energy_sum.append(x)
    # print(ch_energy_sum)

    ch_energy_sum = np.transpose(np.array([ch_energy_sum]))
    ev_pump_energy_sum = np.transpose(np.array([ev_pump_energy_sum]))
    cd_pump_energy_sum = np.transpose(np.array([cd_pump_energy_sum]))
    ct_energy_sum = np.transpose(np.array([ct_energy_sum]))

    ch_energy = np.transpose(np.array(ch_energy))
    ev_pump_energy = np.transpose(np.array(ev_pump_energy))
    cd_pump_energy = np.transpose(np.array(cd_pump_energy))
    ct_energy = np.transpose(np.array(ct_energy))




    TimeS = TimeS[indesame]
    # print(TimeS)
    TimeS = np.transpose(np.array([TimeS]))

    ch_energy_sum = ch_energy_sum[indesame]
    ev_pump_energy_sum = ev_pump_energy_sum[indesame]
    cd_pump_energy_sum = cd_pump_energy_sum[indesame]
    ct_energy_sum = ct_energy_sum[indesame]

    ch_energy = ch_energy[indesame]
    ev_pump_energy = ev_pump_energy[indesame]
    cd_pump_energy = cd_pump_energy[indesame]
    ct_energy = ct_energy[indesame]

    for i in range(len(cd_pump_energy_sum)):
        plant_energy.append(cd_pump_energy_sum[i] + ch_energy_sum[i] + ev_pump_energy_sum[i] + ct_energy_sum[i])

    print('energy end')

    return TimeS, plant_energy,ch_energy_sum,ev_pump_energy_sum,cd_pump_energy_sum,ct_energy_sum,ch_energy,ev_pump_energy,cd_pump_energy,ct_energy
#sheet==load, chille, ev_pump, cd_pump, coolingTower,
def meter_data(CCS_filepath,CHU_filepath,CWP_filepath,CHWP_filepath,CTW_filepath ):

    xl = pd.ExcelFile(CCS_filepath)

    df_CCS = xl.parse()
    # df_CCS = pd.read_excel(CCS_filepath)
    np_time = np.array(df_CCS['Time'])
    np_load = np.array(df_CCS['CCS_LastHourAccCool'])

    xl = pd.ExcelFile(CHU_filepath)

    np_ch_ev_t_out = []
    np_ch_ev_t_in = []
    np_ch_cd_t_out = []
    np_ch_cd_t_in = []

    for sheetname in xl.sheet_names:
        df_CHU = xl.parse(sheet_name=sheetname)
        np_ch_ev_t_out.append(list(df_CHU['CHU_ChilledWaterOutTemp']))
        np_ch_ev_t_in.append(list(df_CHU['CHU_ChilledWaterInTemp']))
        np_ch_cd_t_out.append(list(df_CHU['CHU_CoolingWaterOutTemp']))
        np_ch_cd_t_in.append(list(df_CHU['CHU_CoolingWaterInTemp']))
    np_ch_ev_t_out = np.transpose(np.array(np_ch_ev_t_out))
    np_ch_ev_t_in = np.transpose(np.array(np_ch_ev_t_in))
    np_ch_cd_t_out = np.transpose(np.array(np_ch_cd_t_out))
    np_ch_cd_t_in = np.transpose(np.array(np_ch_cd_t_in))

    print('CHU')


    xl = pd.ExcelFile(CHWP_filepath)

    np_cd_pump_fq = []
    for sheetname in xl.sheet_names:
        df_CHWP = xl.parse(sheet_name=sheetname)
        np_cd_pump_fq.append(list(df_CHWP['CHWP_Freq']))
    np_cd_pump_fq = np.transpose(np.array(np_cd_pump_fq))

    print('CHWP')


    xl = pd.ExcelFile(CWP_filepath)

    np_ev_pump_fq = []
    for sheetname in xl.sheet_names:
        df_CWP = xl.parse(sheet_name=sheetname)
        np_ev_pump_fq.append(list(df_CWP['CWP_Freq']))
    np_ev_pump_fq = np.transpose(np.array(np_ev_pump_fq))

    print('CWP')


    xl = pd.ExcelFile(CTW_filepath)

    np_ct_fq = []
    for sheetname in xl.sheet_names:
        df_CTW = xl.parse(sheet_name=sheetname)
        np_ct_fq.append(list(df_CTW['CTW_FanFreq']))
    np_ct_fq = np.transpose(np.array(np_ct_fq))


    print('CTW')


    indext = np.where(np.array([np_time[i][-5:-3] for i in range(len(np_time))]) == '00')
    # Times = np_time[indext]
    np_ch_ev_t_out = np_ch_ev_t_out[indext]
    np_ch_ev_t_in = np_ch_ev_t_in[indext]
    np_ch_cd_t_out = np_ch_cd_t_out[indext]
    np_ch_cd_t_in = np_ch_cd_t_in[indext]
    np_ev_pump_fq = np_ev_pump_fq[indext]
    np_cd_pump_fq = np_cd_pump_fq[indext]
    np_ct_fq = np_ct_fq[indext]
    np_load = [0.0] + [np.mean(np_load[i:i+4]) for i in indext[0]]
    np_load = np_load[:-1]
    np_load = np.transpose(np.array([np_load]))
    np_time = np.transpose(np.array([np_time[indext]]))


#sheet = weather

    return np_time, np_load, np_ch_ev_t_out, np_ch_ev_t_in, np_ch_cd_t_out,\
           np_ch_cd_t_in,np_ev_pump_fq,np_cd_pump_fq,np_ct_fq
# weather
def data_weather(filepath,CCS_filepath):

    dwhm = pd.read_excel(filepath)

    np_time = processing_time(CCS_filepath)

    # dwhm = pd.read_excel(weather_filepath)
    timess = np.array(dwhm['time'])
    times = []
    for i in timess:
        timeArray = time.strptime(i, "%Y/%m/%d %H:%M:%S")
        timeStamp = int(time.mktime(timeArray))
        timeStamp = round(timeStamp / 3600)
        timeStamp = timeStamp * 3600
        times.append(datetime.datetime.fromtimestamp(timeStamp).strftime('%Y/%m/%d %H:%M:%S'))


    temperature = np.array(dwhm['temperature'])
    humidity = np.array(dwhm['humidity'])
    Twet = []
    Hour = []
    day = []
    dayType = []

    temperatures = []
    humiditys = []


    np_time2 = []
    TimeS2=[]
    for np_times in np_time:
        np_time2.append(time_convert(np_times))
    for j in times:
        TimeS2.append(time_convert_weather(j))
    sametime = [x for x in np_time2 if x in TimeS2]
    indesame = []
    for i in sametime:
        indesame.append(TimeS2.index(i))
    temperatures=temperature[indesame]
    humiditys=humidity[indesame]


    for z in range(len(temperatures)):
        Twet.append(temperatures[z] * math.atan(0.151977 * math.pow((humiditys[z] + 8.313659), 0.5)) +
                    math.atan(temperatures[z] + humiditys[z]) - math.atan(humiditys[z] - 1.676331) +
                    0.00391838 * math.pow(humiditys[z], 1.5) * math.atan(0.023101 * humiditys[z]) - 4.686035)

    for i in range(len(sametime)):
        Hour.append(sametime[i][-2:])

    for i in range(len(sametime)):
        day_s = datetime.datetime.strptime(sametime[i], '%Y-%m-%d %H')
        day.append(day_s.weekday() + 1)
    for i in range(len(sametime)):
        if (day[i] >= 6):
            dayType.append(1)
        else:
            dayType.append(0)
    temperatures = np.transpose(np.array([temperatures]))
    humiditys = np.transpose(np.array([humiditys]))
    weather_time = np.transpose(np.array([sametime]))
    Twet = np.transpose(np.array([Twet]))
    Hour = np.transpose(np.array([Hour]))
    day = np.transpose(np.array([day]))
    dayType = np.transpose(np.array([dayType]))

    return weather_time, temperatures, humiditys, Twet, Hour, day, dayType
if __name__ == "__main__":

    projectId = "Pj4413030001"

    start1 = time.time()

    Tindoor_filepaths = project_path[projectId] + '室内温度.xlsx'

    CCS_filepath = project_path[projectId] +'CCS.xlsx'
    CHU_filepath = project_path[projectId] +'CHU.xlsx'
    CHWP_filepath = project_path[projectId] +'CHWP.xlsx'
    CWP_filepath = project_path[projectId] +'CWP.xlsx'
    CTW_filepath =project_path[projectId] + 'CTW.xlsx'

    weather_filepath =project_path[projectId] + 'weather.xlsx'
    Tindoor_filepath = project_path[projectId] +'energy.xls'

    startin = time.time()
    new_df = data_indoor(Tindoor_filepaths, CCS_filepath)
    endin = time.time()
    print('返回完data_indoor数据所用时间:', (endin - startin))

    startin = time.time()
    DateTime, plant_energy,ch_energy_sum,ev_pump_energy_sum,cd_pump_energy_sum,ct_energy_sum,ch_energy,ev_pump_energy,cd_pump_energy,ct_energy = data_energy(Tindoor_filepath,CCS_filepath)
    endin = time.time()
    print('返回完data_energy数据所用时间:', (endin - startin))
    startin = time.time()
    DateTimes, temperatures, humiditys, Twet, Hour, day, dayType= data_weather(weather_filepath,CCS_filepath)
    endin = time.time()
    print('返回完data_weather数据所用时间:', (endin - startin))
    startin = time.time()
    np_time, np_load, np_ch_ev_t_out, np_ch_ev_t_in, np_ch_cd_t_out, np_ch_cd_t_in, \
    np_ev_pump_fq,np_cd_pump_fq, np_ct_fq  = meter_data(CCS_filepath,CHU_filepath,CWP_filepath,CHWP_filepath,CTW_filepath)
    endin = time.time()
    print('返回完meter_data数据所用时间:', (endin - startin))
    end1 = time.time()
    # save processed data
    print('返回完所有数据所用时间:',(end1-start1))

    start2 = time.time()
    writer = pd.ExcelWriter(project_name[projectId] + '_Plant_all.xlsx')

    new_df.to_excel(writer, sheet_name='Indoor', index=False)

    columns = ['DateTime','plant_load']
    df_load = pd.DataFrame(np.hstack((np_time, np_load)), columns=columns)
    df_load.to_excel(writer,sheet_name='load', index = False)

    columns = ['DateTime'] + ['ch'+str(i)+'_ev_t_out' for i in range(np.shape(np_ch_ev_t_out)[1])]
    columns = columns + ['ch' + str(i) + '_ev_t_in' for i in range(np.shape(np_ch_ev_t_in)[1])]
    columns = columns + ['ch' + str(i) + '_cd_t_out' for i in range(np.shape(np_ch_cd_t_out)[1])]
    columns = columns + ['ch' + str(i) + '_cd_t_in' for i in range(np.shape(np_ch_cd_t_in)[1])]
    df_chiller = pd.DataFrame(np.hstack((np_time, np_ch_ev_t_out,np_ch_ev_t_in,np_ch_cd_t_out,np_ch_cd_t_in)),
                              columns=columns)
    df_chiller.to_excel(writer, sheet_name='chiller', index = False)

    columns = ['DateTime'] + ['ev_pump' + str(i) + '_fq' for i in range(np.shape(np_ev_pump_fq)[1])]
    df_ev_pump = pd.DataFrame(np.hstack((np_time, np_ev_pump_fq)), columns=columns)
    df_ev_pump.to_excel(writer, sheet_name='ev_pump', index = False)

    columns = ['DateTime'] + ['cd_pump' + str(i) + '_fq' for i in range(np.shape(np_cd_pump_fq)[1])]
    df_cd_pump = pd.DataFrame(np.hstack((np_time, np_cd_pump_fq)), columns=columns)
    df_cd_pump.to_excel(writer, sheet_name='cd_pump', index = False)

    columns = ['DateTime'] + ['ct' + str(i) + '_fq' for i in range(np.shape(np_ct_fq)[1])]
    df_ct = pd.DataFrame(np.hstack((np_time, np_ct_fq)), columns=columns)
    df_ct.to_excel(writer, sheet_name='coolingTower', index = False)


    columns = ['DateTime', 'Tout ', 'Hout', 'Twet', 'Hour', 'day', 'dayType']
    df_load = pd.DataFrame(np.hstack((DateTimes, temperatures, humiditys, Twet, Hour, day, dayType)), columns=columns)
    df_load.to_excel(writer, sheet_name='weather', index=False)
    #
    columns = ['DateTime', 'plant_energy', 'ch_energy', 'ev_pump_energy', 'cd_pump_energy', 'ct_energy']
    columns = columns + ['ch' + str(i + 1) + '_energy' for i in range(np.shape(ch_energy)[1])]
    columns = columns + ['ev_pump' + str(i + 1) + '_energy' for i in range(np.shape(ev_pump_energy)[1])]
    columns = columns + ['cd_pump' + str(i + 1) + '_energy' for i in range(np.shape(cd_pump_energy)[1])]
    columns = columns + ['ct' + str(i + 1) + '_energy' for i in range(np.shape(ct_energy)[1])]
    df_energy = pd.DataFrame(np.hstack((DateTime, plant_energy, ch_energy_sum, ev_pump_energy_sum, cd_pump_energy_sum,
                                        ct_energy_sum, ch_energy, ev_pump_energy, cd_pump_energy, ct_energy)),
                             columns=columns)
    df_energy.to_excel(writer, sheet_name='energy', index=False)

    #
    #
    writer.close()
    end2 = time.time()
    print('读入数据所用时间:', (end2 - start2))
    print('所用总时间:', (end2 - start1))
    print("end")
