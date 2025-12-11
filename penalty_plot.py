import numpy as np
import matplotlib.pyplot as plt

def soc_violation_penalty(missing_soc):
    x0 = 0.29229767
    k = 15

    base = -1000 / (1 + np.exp(-k * (missing_soc - x0))) + 1
    zero_point = -1000 / (1 + np.exp(-k * (0 - x0))) + 1

    penalty = base - zero_point
    return penalty

missing_soc_vals = np.linspace(0, 1, 500)
penalties = soc_violation_penalty(missing_soc_vals)

plt.figure(figsize=(8,5))
plt.plot(missing_soc_vals, penalties, color='blue')
plt.xlabel('Missing SOC')
plt.ylabel('Penalty')
plt.title('SOC Violation')
plt.grid(True)
plt.legend()
plt.show()
