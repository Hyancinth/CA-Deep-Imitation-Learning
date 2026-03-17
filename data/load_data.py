import h5py
from collections import defaultdict

def load_data_from_file(file_path):
    """
    Load data from h5 file and return it as a nested dictionary:
    { run_i: {dataset_name: numpy_array, ...}, ... }
    """
    data = {}
    with h5py.File(file_path, 'r') as f:
        for run in f.keys():  # 'run_0', 'run_1',...
            data[run] = {}
            for dataset in f[run].keys():  # 'min_dist_obstacle_link_1',...
                dset = f[run][dataset]
                if dset.shape == (): # handle scalar datasets
                    data[run][dataset] = dset[()]
                else:
                    data[run][dataset] = dset[:] 
    # print u1 dataset for run_0 to verify it was loaded correctly
    if 'run_0' in data and 'u1' in data['run_0']:
        print("u1 dataset for run_0:", data['run_0']['u1'])
    return data

if __name__ == "__main__":
    file_path = "model/data/test_data_1.h5"
    data = load_data_from_file(file_path)
    
    for run, datasets in data.items():
        print(run)
        for name, array in datasets.items():
            print(f"  {name}: {array}")