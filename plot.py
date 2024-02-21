import matplotlib.pyplot as plt

# Data for the first and second figures
matrix_dimensions = [1024, 2048, 4096, 8192]
cpu_times_matrix = [0.0043, 0.0340, 0.2848, 2.1163]
tee_times_matrix = [4.6082, 1.2718, 8.3347, 71.4296]

input_dimensions = ['28*28*512', '56*56*256', '112*112*128','224*224*64']
cpu_times_convolution = [0.0551,0.0282, 0.0271, 0.0372]
tee_times_convolution = [0.0908,0.0904,0.1329,0.2320]

# Data for the third and fourth figures (updated data)
batch_sizes = [1, 2, 4, 6, 8]
trustfl_times = [38.66, 39.16, 57.11, 67.47, 88.09]
goten_times = [8.2, 11.9, 26.42, 43.55, 62.62]
our_protocol_times_new = [2.98, 4.24, 9.07, 15.92, 21.39]

# 1x4 layout
plt.figure(figsize=(24, 6), dpi=800)

# Matrix Production
plt.subplot(1, 4, 1)
plt.plot(matrix_dimensions, cpu_times_matrix, '-o', color='red', label='CPU')
plt.plot(matrix_dimensions, tee_times_matrix, '-o', color='orange', label='TEE')
plt.xlabel('Matrix Dimension')
plt.ylabel('Time (s)')
plt.grid(False)
plt.legend()

# Convolution
plt.subplot(1, 4, 2)
plt.plot(input_dimensions, cpu_times_convolution, '-o', color='red', label='CPU')
plt.plot(input_dimensions, tee_times_convolution, '-o', color='orange', label='TEE')
plt.xlabel('Input Dimension')
plt.ylabel('Time (s)')
plt.grid(False)
plt.legend()

# Updated Data
plt.subplot(1, 4, 3)
plt.plot(batch_sizes, our_protocol_times_new, '-o', color='red', label='Our Protocol')
plt.plot(batch_sizes, trustfl_times, '-o', color='orange', label='TrustFL')
plt.xlabel('Batch Size')
plt.ylabel('Time (s)')
plt.grid(False)
plt.legend()

plt.subplot(1, 4, 4)
plt.plot(batch_sizes, our_protocol_times_new, '-o', color='red', label='Our Protocol')
plt.plot(batch_sizes, goten_times, '-o', color='orange', label='Goten')
plt.xlabel('Batch Size')
plt.ylabel('Time (s)')
plt.grid(False)
plt.legend()

plt.savefig('comparison_chart_1x4.png', dpi=800)
plt.close()

# 2x2 layout
plt.figure(figsize=(12, 12), dpi=800)

# Matrix Production
plt.subplot(2, 2, 1)
plt.plot(matrix_dimensions, cpu_times_matrix, '-o', color='red', label='CPU')
plt.plot(matrix_dimensions, tee_times_matrix, '-o', color='orange', label='TEE')
plt.xlabel('Matrix Dimension')
plt.ylabel('Time (s)')
plt.grid(False)
plt.legend()

# Convolution
plt.subplot(2, 2, 2)
plt.plot(input_dimensions, cpu_times_convolution, '-o', color='red', label='CPU')
plt.plot(input_dimensions, tee_times_convolution, '-o', color='orange', label='TEE')
plt.xlabel('Input Dimension')
plt.ylabel('Time (s)')
plt.grid(False)
plt.legend()

# Updated Data
plt.subplot(2, 2, 3)
plt.plot(batch_sizes, our_protocol_times_new, '-o', color='red', label='Our Protocol')
plt.plot(batch_sizes, trustfl_times, '-o', color='orange', label='TrustFL')
plt.xlabel('Batch Size')
plt.ylabel('Time (s)')
plt.grid(False)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(batch_sizes, our_protocol_times_new, '-o', color='red', label='Our Protocol')
plt.plot(batch_sizes, goten_times, '-o', color='orange', label='Goten')
plt.xlabel('Batch Size')
plt.ylabel('Time (s)')
plt.grid(False)
plt.legend()

plt.savefig('comparison_chart_2x2.png', dpi=800)
plt
