import pandas as pd

original_info = pd.read_csv("df_train_combined.csv")
test_info = pd.read_csv("for_test.csv")

concated_info = pd.concat([original_info, test_info], sort=False)
concated_info['index'] = [i for i in range(len(concated_info))]

concated_info.to_csv('./test.csv', index=False)