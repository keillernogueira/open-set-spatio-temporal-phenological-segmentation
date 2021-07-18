import numpy as np
import imageio

a = imageio.imread('C:\\Users\\keill\\Desktop\\images.jpg')
print(a.shape)

arr = a

arr = np.asarray(arr)
print(arr.shape)

# fliplr = np.fliplr(arr)
# flipud = np.flipud(arr)
flipud = np.flip(arr, axis=0)
fliplr = np.flip(arr, axis=1)

rot1 = np.rot90(arr, 1, (0, 1))
rot2 = np.rot90(arr, 2, (0, 1))
rot3 = np.rot90(arr, 3, (0, 1))

imageio.imwrite('C:\\Users\\keill\\Desktop\\fliplr.png', fliplr)
imageio.imwrite('C:\\Users\\keill\\Desktop\\flipud.png', flipud)
imageio.imwrite('C:\\Users\\keill\\Desktop\\rot1.png', rot1)
imageio.imwrite('C:\\Users\\keill\\Desktop\\rot2.png', rot2)
imageio.imwrite('C:\\Users\\keill\\Desktop\\rot3.png', rot3)


# multiple images
# arr = []
# arr.append(a)
# arr.append(a)
#
# arr = np.asarray(arr)
# print(arr.shape)
#
# # fliplr = np.fliplr(arr)
# # flipud = np.flipud(arr)
# flipud = np.flip(arr, axis=1)
# fliplr = np.flip(arr, axis=2)
#
# rot1 = np.rot90(arr, 1, (1, 2))
# rot2 = np.rot90(arr, 2, (1, 2))
# rot3 = np.rot90(arr, 3, (1, 2))
#
# for i in range(2):
#     imageio.imwrite('C:\\Users\\keill\\Desktop\\fliplr_' + str(i) + '.png', fliplr[i])
#     imageio.imwrite('C:\\Users\\keill\\Desktop\\flipud_' + str(i) + '.png', flipud[i])
#     imageio.imwrite('C:\\Users\\keill\\Desktop\\rot1_' + str(i) + '.png', rot1[i])
#     imageio.imwrite('C:\\Users\\keill\\Desktop\\rot2_' + str(i) + '.png', rot2[i])
#     imageio.imwrite('C:\\Users\\keill\\Desktop\\rot3_' + str(i) + '.png', rot3[i])

