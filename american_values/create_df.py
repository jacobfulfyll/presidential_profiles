import pandas as pd
import psycopg2 as pg2
from sqlalchemy import create_engine

df1 = pd.read_csv('aws_dfs/rows_0_1000.csv')
df2 = pd.read_csv('aws_dfs/rows_1000_2000.csv')
df3 = pd.read_csv('aws_dfs/rows_2000_2500.csv')
df4 = pd.read_csv('aws_dfs/rows_2500_3000.csv')
df5 = pd.read_csv('aws_dfs/rows_3000_3500.csv')
df6 = pd.read_csv('aws_dfs/rows_3500_4000.csv')
df7 = pd.read_csv('aws_dfs/rows_4000_5000.csv')
df9 = pd.read_csv('aws_dfs/rows_5000_5500.csv')
df10 = pd.read_csv('aws_dfs/rows_5500_6000.csv')
df11 = pd.read_csv('aws_dfs/rows_6000_6500.csv')
df12 = pd.read_csv('aws_dfs/rows_6500_7000.csv')
df13 = pd.read_csv('aws_dfs/rows_7000_7500.csv')
df14 = pd.read_csv('aws_dfs/rows_7500_8000.csv')
df15 = pd.read_csv('aws_dfs/rows_8000_8500.csv')
df16 = pd.read_csv('aws_dfs/rows_8500_9000.csv')
df17 = pd.read_csv('aws_dfs/rows_9000_9500.csv')
df18 = pd.read_csv('aws_dfs/rows_9500_10000.csv')
df19 = pd.read_csv('aws_dfs/rows_10000_10500.csv')
df20 = pd.read_csv('aws_dfs/rows_10500_11000.csv')
df21 = pd.read_csv('aws_dfs/rows_11000_11500.csv')
df22 = pd.read_csv('aws_dfs/rows_11500_12000.csv')
df23 = pd.read_csv('aws_dfs/rows_12000_12500.csv')
df24 = pd.read_csv('aws_dfs/rows_12500_13000.csv')
df25 = pd.read_csv('aws_dfs/rows_13000_13500.csv')
df26 = pd.read_csv('aws_dfs/rows_13500_14000.csv')
df27 = pd.read_csv('aws_dfs/rows_14000_14500.csv')
df28 = pd.read_csv('aws_dfs/rows_14500_15000.csv')
df29 = pd.read_csv('aws_dfs/rows_15000_15500.csv')
df30 = pd.read_csv('aws_dfs/rows_15500_16000.csv')
df31 = pd.read_csv('aws_dfs/rows_16000_16500.csv')
df32 = pd.read_csv('aws_dfs/rows_16500_17000.csv')
df33 = pd.read_csv('aws_dfs/rows_17000_17500.csv')
df34 = pd.read_csv('aws_dfs/rows_17500_18000.csv')
df35 = pd.read_csv('aws_dfs/rows_18000_18500.csv')
df36 = pd.read_csv('aws_dfs/rows_18500_19000.csv')
df37 = pd.read_csv('aws_dfs/rows_19000_19500.csv')
df38 = pd.read_csv('aws_dfs/rows_19500_20000.csv')
df39 = pd.read_csv('aws_dfs/rows_20000_20500.csv')
df40 = pd.read_csv('aws_dfs/rows_20500_21000.csv')
df41 = pd.read_csv('aws_dfs/rows_21000_21500.csv')
df42 = pd.read_csv('aws_dfs/rows_21700_21900.csv')
df43 = pd.read_csv('aws_dfs/rows_21900_22000.csv')
df44 = pd.read_csv('aws_dfs/rows_22000_22500.csv')
df45 = pd.read_csv('aws_dfs/rows_22500_23000.csv')
df46 = pd.read_csv('aws_dfs/rows_23000_23500.csv')
df47 = pd.read_csv('aws_dfs/rows_23500_24000.csv')
df48 = pd.read_csv('aws_dfs/rows_24000_24500.csv')
df49 = pd.read_csv('aws_dfs/rows_24500_25000.csv')
df49 = pd.read_csv('aws_dfs/rows_25000_25500.csv')
df50 = pd.read_csv('aws_dfs/rows_25500_26000.csv')
df51 = pd.read_csv('aws_dfs/rows_27500_27700.csv')
df52 = pd.read_csv('aws_dfs/rows_27700_28000.csv')
df53 = pd.read_csv('aws_dfs/rows_28000_28200.csv')
df54 = pd.read_csv('aws_dfs/rows_28200_28400.csv')
df55 = pd.read_csv('aws_dfs/rows_28400_28500.csv')
df56 = pd.read_csv('aws_dfs/rows_28500_28600.csv')
df57 = pd.read_csv('aws_dfs/rows_28600_28800.csv')
df58 = pd.read_csv('aws_dfs/rows_28800_29000.csv')
df59 = pd.read_csv('aws_dfs/rows_29000_29200.csv')
df61 = pd.read_csv('aws_dfs/rows_29200_29400.csv')
df62 = pd.read_csv('aws_dfs/rows_29400_29600.csv')
df63 = pd.read_csv('aws_dfs/rows_29600_29800.csv')
df64 = pd.read_csv('aws_dfs/rows_29800_30000.csv')
df65 = pd.read_csv('aws_dfs/rows_30000_30100.csv')
df66 = pd.read_csv('aws_dfs/rows_30100_30200.csv')
df67 = pd.read_csv('aws_dfs/rows_30200_30300.csv')
df68 = pd.read_csv('aws_dfs/rows_30300_30400.csv')
df69 = pd.read_csv('aws_dfs/rows_30400_30500.csv')
df70 = pd.read_csv('aws_dfs/rows_30500_30600.csv')
df71 = pd.read_csv('aws_dfs/rows_30600_30700.csv')
df72 = pd.read_csv('aws_dfs/rows_30700_30800.csv')
df73 = pd.read_csv('aws_dfs/rows_30800_30900.csv')
df74 = pd.read_csv('aws_dfs/rows_30900_31000.csv')
df75 = pd.read_csv('aws_dfs/rows_31000_31200.csv')
df76 = pd.read_csv('aws_dfs/rows_31200_31400.csv')
df77 = pd.read_csv('aws_dfs/rows_31400_31600.csv')
df78 = pd.read_csv('aws_dfs/rows_31600_31800.csv')
df79 = pd.read_csv('aws_dfs/rows_31800_32000.csv')
df80 = pd.read_csv('aws_dfs/rows_32000_32200.csv')
df81 = pd.read_csv('aws_dfs/rows_32200_32400.csv')
df82 = pd.read_csv('aws_dfs/rows_32400_32600.csv')
df83 = pd.read_csv('aws_dfs/rows_32600_32800.csv')
df84 = pd.read_csv('aws_dfs/rows_32800_33000.csv')
df85 = pd.read_csv('aws_dfs/rows_33000_33200.csv')
df86 = pd.read_csv('aws_dfs/rows_33200_33400.csv')
df87 = pd.read_csv('aws_dfs/rows_33400_33600.csv')
df88 = pd.read_csv('aws_dfs/rows_33600_33800.csv')
df89 = pd.read_csv('aws_dfs/rows_33800_34000.csv')
df90 = pd.read_csv('aws_dfs/rows_34000_34200.csv')
df91 = pd.read_csv('aws_dfs/rows_34200_34400.csv')
df92 = pd.read_csv('aws_dfs/rows_34400_34600.csv')
df93 = pd.read_csv('aws_dfs/rows_34600_34800.csv')
df94 = pd.read_csv('aws_dfs/rows_34800_35000.csv')
df95 = pd.read_csv('aws_dfs/rows_35000_35200.csv')
df96 = pd.read_csv('aws_dfs/rows_35200_35400.csv')
df97 = pd.read_csv('aws_dfs/rows_35400_35600.csv')
df98 = pd.read_csv('aws_dfs/rows_35600_35800.csv')
df99 = pd.read_csv('aws_dfs/rows_35800_36000.csv')
df100 = pd.read_csv('aws_dfs/rows_36000_36200.csv')
df101 = pd.read_csv('aws_dfs/rows_36200_36400.csv')
df102 = pd.read_csv('aws_dfs/rows_36400_36600.csv')
df103 = pd.read_csv('aws_dfs/rows_36600_36800.csv')
df104 = pd.read_csv('aws_dfs/rows_36800_37000.csv')
df105 = pd.read_csv('aws_dfs/rows_37000_37200.csv')
df106 = pd.read_csv('aws_dfs/rows_37200_37400.csv')
df107 = pd.read_csv('aws_dfs/rows_37400_37600.csv')
df108 = pd.read_csv('aws_dfs/rows_37600_37800.csv')
df109 = pd.read_csv('aws_dfs/rows_37800_38000.csv')
df110 = pd.read_csv('aws_dfs/rows_38000_38200.csv')
df111 = pd.read_csv('aws_dfs/rows_38200_38400.csv')
df112 = pd.read_csv('aws_dfs/rows_38400_38600.csv')
df113 = pd.read_csv('aws_dfs/rows_38600_38800.csv')
df114 = pd.read_csv('aws_dfs/rows_38800_39000.csv')
df115 = pd.read_csv('aws_dfs/rows_39000_39200.csv')
df116 = pd.read_csv('aws_dfs/rows_39200_39400.csv')
df117 = pd.read_csv('aws_dfs/rows_39400_39600.csv')
df118 = pd.read_csv('aws_dfs/rows_39600_39800.csv')
df119 = pd.read_csv('aws_dfs/rows_39800_40000.csv')
df120 = pd.read_csv('aws_dfs/rows_40000_40200.csv')
df121 = pd.read_csv('aws_dfs/rows_40200_40400.csv')
df122 = pd.read_csv('aws_dfs/rows_40400_40600.csv')
df123 = pd.read_csv('aws_dfs/rows_40600_40800.csv')
df124 = pd.read_csv('aws_dfs/rows_40800_41000.csv')
df125 = pd.read_csv('aws_dfs/rows_41000_41200.csv')
df126 = pd.read_csv('aws_dfs/rows_41200_41400.csv')
df127 = pd.read_csv('aws_dfs/rows_41400_41600.csv')
df128 = pd.read_csv('aws_dfs/rows_41600_41800.csv')
df129 = pd.read_csv('aws_dfs/rows_41800_42000.csv')
df130 = pd.read_csv('aws_dfs/rows_42000_42200.csv')
df131 = pd.read_csv('aws_dfs/rows_42200_42400.csv')
df132 = pd.read_csv('aws_dfs/rows_42400_42600.csv')
df133 = pd.read_csv('aws_dfs/rows_42600_42800.csv')
df134 = pd.read_csv('aws_dfs/rows_42800_43000.csv')
df135 = pd.read_csv('aws_dfs/rows_43000_43200.csv')
df136 = pd.read_csv('aws_dfs/rows_43200_43400.csv')
df137 = pd.read_csv('aws_dfs/rows_43400_43600.csv')
df138 = pd.read_csv('aws_dfs/rows_43600_43800.csv')
df139 = pd.read_csv('aws_dfs/rows_43800_44000.csv')
df140 = pd.read_csv('aws_dfs/rows_44000_44200.csv')
df141 = pd.read_csv('aws_dfs/rows_44200_44400.csv')
df142 = pd.read_csv('aws_dfs/rows_44400_44600.csv')
df143 = pd.read_csv('aws_dfs/rows_44600_44800.csv')
df144 = pd.read_csv('aws_dfs/rows_44800_45000.csv')
df145 = pd.read_csv('aws_dfs/rows_45000_45200.csv')
df146 = pd.read_csv('aws_dfs/rows_45200_45400.csv')
df147 = pd.read_csv('aws_dfs/rows_45400_45600.csv')
df148 = pd.read_csv('aws_dfs/rows_45600_45800.csv')
df149 = pd.read_csv('aws_dfs/rows_45800_46000.csv')
df150 = pd.read_csv('aws_dfs/rows_46000_46200.csv')
df151 = pd.read_csv('aws_dfs/rows_46200_46400.csv')
df152 = pd.read_csv('aws_dfs/rows_46400_46600.csv')
df153 = pd.read_csv('aws_dfs/rows_46600_46800.csv')
df154 = pd.read_csv('aws_dfs/rows_46800_47000.csv')
df155 = pd.read_csv('aws_dfs/rows_47000_47200.csv')
df156 = pd.read_csv('aws_dfs/rows_47200_47400.csv')
df157 = pd.read_csv('aws_dfs/rows_47400_47600.csv')
df158 = pd.read_csv('aws_dfs/rows_47600_47800.csv')
df159 = pd.read_csv('aws_dfs/rows_47800_48000.csv')
df160 = pd.read_csv('aws_dfs/rows_48000_48200.csv')
df161 = pd.read_csv('aws_dfs/rows_48200_48400.csv')
df162 = pd.read_csv('aws_dfs/rows_48400_48600.csv')
df163 = pd.read_csv('aws_dfs/rows_48600_48800.csv')
df164 = pd.read_csv('aws_dfs/rows_48800_49000.csv')
df165 = pd.read_csv('aws_dfs/rows_49000_49200.csv')
df166 = pd.read_csv('aws_dfs/rows_49200_49400.csv')
df167 = pd.read_csv('aws_dfs/rows_49400_49600.csv')
df168 = pd.read_csv('aws_dfs/rows_49600_49800.csv')
df169 = pd.read_csv('aws_dfs/rows_49800_on.csv')

dfs = [df1, df2, df3, df4, df5, df6, df7, df9, df10,
df11,
df12,
df13,
df14,
df15,
df16,
df17,
df18,
df19,
df20,
df21,
df22,
df23,
df24,
df25,
df26,
df27,
df28,
df29,
df30,
df31,
df32,
df33,
df34,
df35,
df36,
df37,
df38,
df39,
df40,
df41,
df42,
df43,
df44,
df45,
df46,
df47,
df48,
df49,
df49,
df50,
df51,
df52,
df53,
df54,
df55,
df56,
df57,
df58,
df59,
df61,
df62,
df63,
df64,
df65,
df66,
df67,
df68,
df69,
df70,
df71,
df72,
df73,
df74,
df75,
df76,
df77,
df78,
df79,
df80,
df81,
df82,
df83,
df84,
df85,
df86,
df87,
df88,
df89,
df90,
df91,
df92,
df93,
df94,
df95,
df96,
df97,
df98,
df99,
df100,
df101,
df102,
df103,
df104,
df105,
df106,
df107,
df108,
df109,
df110,
df111,
df112,
df113,
df114,
df115,
df116,
df117,
df118,
df119,
df120,
df121,
df122,
df123,
df124,
df125,
df126,
df127,
df128,
df129,
df130,
df131,
df132,
df133,
df134,
df135,
df136,
df137,
df138,
df139,
df140,
df141,
df142,
df143,
df144,
df145,
df146,
df147,
df148,
df149,
df150,
df151,
df152,
df153,
df154,
df155,
df156,
df157,
df158,
df159,
df160,
df161,
df162,
df163,
df164,
df165,
df166,
df167,
df168,
df169]

conn = pg2.connect(dbname = 'postgres', host = "localhost")
conn.autocommit = True
engine = create_engine('postgresql+psycopg2://owner:Fulfyll@localhost/rhetoric_capstone')
print(len(dfs))
for df in dfs:
    print(len(df))
    df.to_sql("presidesnts_words", con = engine, if_exists= "append", index=False)
conn.close()