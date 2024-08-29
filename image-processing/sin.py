import numpy as np
import matplotlib.pyplot as plt

# Tạo dãy số từ 0 đến 2*pi với bước nhỏ
x = np.linspace(0, 10, 1000)
# Tính giá trị sin cho từng điểm trên x
def sum(x):
    f1 = 5*np.sin(4*x)
    f2 = 2.5*np.sin(7*x)
    f3 = 0.5*np.sin(14*x)
    f4=f1+f2+f3
    return f1,f2,f3,f4

f1,f2,f3,f4=sum(x)

# Tạo subplot 2x2 và vẽ từng đồ thị sóng sin
plt.subplot(2, 2, 1)
plt.plot(x, f1)
plt.title('f1')

plt.subplot(2, 2, 2)
plt.plot(x, f2)
plt.title('f2')

plt.subplot(2, 2, 3)
plt.plot(x, f3)
plt.title('f3')

plt.subplot(2, 2, 4)
plt.plot(x, f4)
plt.title('f4')

plt.suptitle('Đồ thị các sóng sin')
plt.show()