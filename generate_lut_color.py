import numpy as np

def temperature_to_color(temp):
    t = temp / 100.0
    if temp<1000.0:
        return 0,0,0
    
    if t <= 66.0:
        r = 1
    else:
        r = 1.292936186 * np.pow(t - 60.0, -0.1332047592)
        r = np.clip(r,0,1)

    if t <= 66.0: 
        g = 0.39008157876 * np.log(t) - 0.631841444
        g = np.clip(g,0,1)
    else:
        g = 1.129890861 * np.pow(t - 60.0, -0.0755148492)
        g = np.clip(g,0,1)
    
    if t >= 66.0:
        b = 1.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 0.543206789 * np.log(t - 10.0) - 1.196254089
        b = np.clip(b,0,1)
    

    return r, g, b
def planck_discrete_color(temp):
    """
    使用离散三波长普朗克公式计算黑体辐射颜色（红端更红，蓝端更蓝）
    """
    if temp < 100.0:
        return 0.0, 0.0, 0.0
    
    # 定义 RGB 三原色对应的窄带物理波长（单位：微米）
    # 大佬常用的黄金参数：740nm (极深红), 560nm (绿), 440nm (深蓝)
    # 这三个波长通道之间没有重合，因此色彩饱和度会达到极致
    lam = np.array([0.740, 0.560, 0.440], dtype=np.float32)
    
    # 普朗克第二常数 c2 = h*c/k_b ≈ 14387.76 微米·开尔文
    c2 = 14387.76
    
    # 普朗克公式求三个波长处的本征辐射强度
    # I(lam, T) = 1 / (lam^5 * (exp(c2 / (lam * T)) - 1))
    denom = (lam**5) * (np.exp(c2 / (lam * temp)) - 1.0)
    intensity = 1.0 / denom
    
    # 归一化色度：将最大通道的值强行拉伸到 1.0
    # 这样可以完美分离“颜色”与“亮度”，让 LUT 只记录色相
    max_val = np.max(intensity)
    if max_val > 0.0:
        r, g, b = intensity / max_val
    else:
        r, g, b = 0.0, 0.0, 0.0
        
    return r, g, b

u = np.empty((1,2000,3),dtype=np.float32)
for i in range(2000):
    t = i*9.75+510
    # r,g,b=temperature_to_color(t)
    r,g,b = planck_discrete_color(t)
    u[0,i,:]=(r,g,b)
np.save('color_lut2.npy', u)
print("LUT 生成完成，形状:", u.shape)