import pandas as pd

df_results = pd.read_csv("test_results_v2.csv")
df_results = df_results.assign(Real=lambda x: x["filepath"].str.split("/").str[2])

df_results = df_results[["Real","prediction"]]
df_results_false = df_results[df_results["prediction"] != df_results["Real"]]
df_results_true = df_results[df_results["prediction"] == df_results["Real"]]
print(df_results_true.shape[0])
print(df_results_false.shape[0])
print("Model doğruluk oranı: " , df_results_true.shape[0]/df_results.shape[0])
print(df_results_false.head(df_results_false.shape[0]))