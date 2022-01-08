from sklearn.model_selection import KFold

kf = KFold(n_splits=4, random_state=0, shuffle=True)
a = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

for i, j in kf.split(a):
    print(i, j)