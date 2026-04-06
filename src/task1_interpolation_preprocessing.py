import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
import numpy as np
from scipy import stats

# 加载自定义字体文件 (中文宋体字体 和 英文字体 Times New Roman)
font_paths = [
    "simsun.ttc", "simsunb.ttf", "SimsunExtG.ttf",  # 宋体字体文件
    "times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"  # Times New Roman字体文件
]
for fp in font_paths:
    font_manager.fontManager.addfont(fp)

# 设置中文字体为 宋体，英文字体为 Times New Roman
plt.rcParams["font.family"] = "SimSun"  # 中文字体为宋体
plt.rcParams["axes.unicode_minus"] = False  # 处理负号显示问题

# 加载数据
file_path = "ais-USV_filled.csv"  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 将 base_date_time 转换为日期时间格式
data['base_date_time'] = pd.to_datetime(data['base_date_time'])

# 插值法填补缺失值（线性插值）
data['longitude'] = data['longitude'].interpolate(method='linear')
data['latitude'] = data['latitude'].interpolate(method='linear')
data['sog'] = data['sog'].interpolate(method='linear')
data['cog'] = data['cog'].interpolate(method='linear')

# 计算SOG和COG的误差（标准差）以及置信区间
sog_std = data['sog'].std()  # 计算SOG的标准差
cog_std = data['cog'].std()  # 计算COG的标准差

# 计算SOG的置信区间（95%）
sog_ci = stats.norm.interval(0.95, loc=data['sog'].mean(), scale=sog_std / np.sqrt(len(data)))

# 计算COG的置信区间（95%）
cog_ci = stats.norm.interval(0.95, loc=data['cog'].mean(), scale=cog_std / np.sqrt(len(data)))

# 平滑方法：滑动平均平滑
data['sog_smoothed'] = data['sog'].rolling(window=5).mean()  # 使用窗口为5的滑动平均进行平滑
data['cog_smoothed'] = data['cog'].rolling(window=5).mean()  # 使用窗口为5的滑动平均进行平滑

# 1. 绘制船舶历史轨迹 (经度 vs 纬度)，并标记开始点和停止点
plt.figure(figsize=(3, 2), dpi=600)  # 设置图片尺寸为3x2，DPI为600
plt.plot(data['longitude'], data['latitude'], label="船舶轨迹", color='blue', linewidth=0.5)  # 更细的线条
plt.scatter(data['longitude'].iloc[0], data['latitude'].iloc[0], color='green', label='起点', zorder=5, s=50, alpha=0.5)  # 起点，减小标记点大小
plt.scatter(data['longitude'].iloc[-1], data['latitude'].iloc[-1], color='red', label='终点', zorder=5, s=50, alpha=0.5)  # 终点
plt.xlabel("经度", fontsize=7.5)
plt.ylabel("纬度", fontsize=7.5)
plt.grid(True, linestyle='--', alpha=0.5)  # 透明度降低
plt.legend(loc='upper right', fontsize=7.5)
plt.tick_params(axis='both', direction='in', labelsize=7.5)  # 刻度朝内，字体为7.5pt
plt.tight_layout()
plt.show()  # 确保图表显示

# 2. 绘制SOG图（带误差条）
plt.figure(figsize=(3, 2), dpi=600)  # 设置图片尺寸为3x2，DPI为600
plt.plot(data['base_date_time'], data['sog_smoothed'], color='lightblue', label='SOG (平滑)', linewidth=0.5, alpha=0.6)  # 更细的线条并增加透明度
plt.fill_between(data['base_date_time'], data['sog_smoothed'] - sog_std, data['sog_smoothed'] + sog_std, color='lightgray', alpha=0.3)  # 误差带透明且细
plt.xlabel('时间', fontsize=7.5)
plt.ylabel('地面速率/节', fontsize=7.5)
plt.xticks(rotation=45, fontsize=7.5)
plt.grid(True, linestyle='--', alpha=0.5)  # 透明度降低
plt.tick_params(axis='both', direction='in', labelsize=7.5)  # 刻度朝内，字体为7.5pt
plt.tight_layout()
plt.show()  # 确保图表显示

# 3. 绘制COG图（带误差条）
plt.figure(figsize=(3, 2), dpi=600)  # 设置图片尺寸为3x2，DPI为600
plt.plot(data['base_date_time'], data['cog_smoothed'], color='lightcoral', label='COG (平滑)', linewidth=0.5, alpha=0.6)  # 更细的线条并增加透明度
plt.fill_between(data['base_date_time'], data['cog_smoothed'] - cog_std, data['cog_smoothed'] + cog_std, color='lightgray', alpha=0.3)  # 误差带透明且细
plt.xlabel('时间', fontsize=7.5)
plt.ylabel('地面航向/度', fontsize=7.5)
plt.xticks(rotation=45, fontsize=7.5)
plt.grid(True, linestyle='--', alpha=0.5)  # 透明度降低
plt.tick_params(axis='both', direction='in', labelsize=7.5)  # 刻度朝内，字体为7.5pt
plt.tight_layout()
plt.show()  # 确保图表显示

# 4. 绘制速度分布直方图 (SOG)，并加上KDE曲线和均值红线
plt.figure(figsize=(3, 2), dpi=600)  # 设置图片尺寸为3x2，DPI为600
sns.histplot(data['sog'], kde=True, color='green', bins=30, stat='density', alpha=0.5, kde_kws={"bw_adjust": 1.5})  # 调整KDE平滑度，透明度降低
mean_sog = data['sog'].mean()
plt.axvline(mean_sog, color='red', linestyle='--', label=f'均值 : {mean_sog:.2f} 节')  # 均值红线
plt.xlabel("地面速率/节", fontsize=7.5)
plt.ylabel("密度", fontsize=7.5)
plt.grid(True, linestyle='--', alpha=0.5)  # 透明度降低
plt.legend(loc='upper right', fontsize=7.5)
plt.tick_params(axis='both', direction='in', labelsize=7.5)  # 刻度朝内，字体为7.5pt
plt.tight_layout()
plt.show()  # 确保图表显示