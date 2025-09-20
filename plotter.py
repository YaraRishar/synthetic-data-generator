import matplotlib.pyplot as plt


def get_mid(arr: list):
    mid = []
    for i in range(len(arr[0])):
        current_elem = 0
        for j in range(len(arr)):
            current_elem += arr[j][i]
        mid.append(round(current_elem / len(arr), 4))
    return mid


iou = [[0.901, 0.9023, 0.897, 0.8963, 0.8841, 0.8676, 0.8767, 0.7176, 0.8311, 0.7954, 0.7769],
       [0.905, 0.9103, 0.8796, 0.8752, 0.8602, 0.8503, 0.8528, 0.8373, 0.8209, 0.7926, 0.7491],
       [0.8991, 0.9028, 0.8949, 0.8971, 0.8725, 0.8877, 0.8742, 0.8499, 0.8597, 0.803, 0.7521]]
loss = [[0.141, 0.132, 0.1452, 0.1417, 0.1573, 0.1794, 0.1651, 0.3953, 0.2329, 0.2811, 0.2985],
        [0.1394, 0.1243, 0.1665, 0.1648, 0.1845, 0.2029, 0.193, 0.2214, 0.2452, 0.2734, 0.345],
        [0.1425, 0.137, 0.1473, 0.1457, 0.1743, 0.1597, 0.1744, 0.1998, 0.193, 0.2719, 0.346]]
# iou = [[0.9063, 0.8995, 0.9026, 0.9153, 0.903, 0.8943, 0.9116, 0.9014, 0.9064, 0.4861, 0.9092],
#        [0.9095, 0.9093, 0.9052, 0.9193, 0.9116, 0.8993, 0.8974, 0.8915, 0.9047, 0.918, 0.9113]]
# loss = [[0.1278, 0.1365, 0.1362, 0.121, 0.1358, 0.143, 0.125, 0.1356, 0.1324, 1.0043, 0.1272],
#         [0.1295, 0.1287, 0.1326, 0.1194, 0.1259, 0.1409, 0.1404, 0.1468, 0.1316, 0.1215, 0.1234]]

data = {"iou": get_mid(iou), "loss": get_mid(loss),
        "synthetic_size": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

print(data)

fig, ax1 = plt.subplots()

ax1.set_xlabel("Доля синтетических данных")
ax1.set_ylabel("Метрика IoU", color="blue")
ax1.plot(data["synthetic_size"], data["iou"], color="blue", marker="o", label="IoU")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.set_ylim(0, 1)

ax2 = ax1.twinx()
ax2.set_ylabel("Loss", color="red")
ax2.plot(data["synthetic_size"], data["loss"], color="red", marker="x", label="Loss")
ax2.tick_params(axis="y", labelcolor="red")
ax2.set_ylim(0, 1)

ax1.grid()
plt.title("Дополнение синтетическими данными")
plt.show()
