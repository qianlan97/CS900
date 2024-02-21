# def compare_text_files(file1_path, file2_path):
#     with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
#         content1 = file1.read()
#         content2 = file2.read()
        
#         if content1 == content2:
#             print("The files are exactly the same.")
#         else:
#             print("The files are different.")

# # Usage example
# file1_path = 'result_outside.txt'
# file2_path = 'result_inside.txt'
# compare_text_files(file1_path, file2_path)

import torch

file1 = 'gradients_outside.pt'
file2 = 'gradients_inside.pt'

gradients_file1 = torch.load(file1)
gradients_file2 = torch.load(file2)

# Compare the keys in the dictionaries
keys_file1 = set(gradients_file1.keys())
keys_file2 = set(gradients_file2.keys())
# Check for keys present in both files
common_keys = keys_file1.intersection(keys_file2)

# Compare the gradients for each common key
for key in common_keys:
    gradients1 = gradients_file1[key]
    gradients2 = gradients_file2[key]
    # print(gradients1, ", ", gradients2)
    # Compare the gradients element-wise
    are_equal = all(torch.allclose(grad1, grad2) for grad1, grad2 in zip(gradients1, gradients2))

    if are_equal:
        print("Gradients for key '{}' are equal.".format(key))
    else:
        print("Gradients for key '{}' are not equal.".format(key))