def column_reader(file_path, col_number):
    """assume file is a csv file with a really large number of colums"""
    f = open(file_path, "r") # file handler
    lines = f.readlines()
    result=[]
    for x in lines:
        result.append(int(x.split(',')[col_number]))
    f.close()
    return result

# df = pd.read_csv('HF.csv', dtype={'x': np.float}, header=None, delimiter=',', engine='python', na_filter=False)

hf_path = '/Users/alex.decastro/development/gitlab/belkin_challenge/data/CSV_OUT/Tagged_Training_02_15_1360915201/HF.csv'

print(column_reader(hf_path, 0))