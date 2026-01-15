import numpy as np
import os


def check_dimensions(data_path):
    errors = []

    if not os.path.exists(data_path):
        print(f"The directory {data_path} does not exist.")
        return

    for root, dirs, files in os.walk(data_path):
        print(files)
        for file in files:
            if 'data.npy' in file:
                print(file)
                image_path = os.path.join(root, file)
                gt_sparse_path = os.path.join(root,  file.replace('data', 'gt_sparse'))
                gt_dense_path = os.path.join(root, file.replace('data', 'gt_alpha'))

                if os.path.exists(gt_sparse_path) and os.path.exists(gt_dense_path):
                    image = np.load(image_path)
                    sparse = np.load(gt_sparse_path)
                    dense = np.load(gt_dense_path)

                    if image.shape != sparse.shape:
                        errors.append((image_path, gt_sparse_path))
                    if image.shape != dense.shape:
                        errors.append((image_path, gt_dense_path))

    if not errors:
        print("All image and label dimensions match.")
    else:
        print("Dimension mismatches found in the following files:")
        for error in errors:
            print(error)


# Replace 'data_path' with the path to your ToothFairy dataset
data_path = '/path/to/toothfairy'
data_path = r'F:\ToothFairy_Dataset\ToothFairy_Dataset\Dataset'
print(data_path)
check_dimensions(data_path)
