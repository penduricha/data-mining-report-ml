# %%
# Nội dung phần báo cáo.
# 1 Mô tả tập dữ liệu và cho biết các thông tin của tập dữ liệu.
# 2 Tiền xử lý dữ liệu.
# 3 Thống kê mô tả và trực quan hóa dữ liệu.
# 4 Xây dựng mô hình Cây quyết định (Decision Tree), dùng GridSearch và Fold xác định tham số.
# 5 Đánh giá mô hình.
# 6 Xây dựng mô hình Losgistic Regression, KNN, Bayes và so sánh với mô hình Cây quyết định.

# %%
# Import thư viện cần thiết
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, recall_score, r2_score, accuracy_score, roc_auc_score, precision_score, classification_report, log_loss

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Thư viện Grid Search CV
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score

# Mô hình so sánh
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# %%
# Dữ liệu đã được chia thành Age, Sex, Blood, Presure and Cholesterol của 200 bệnh nhân.
# Yêu cầu tìm loại thuốc Drug phù hợp cho bệnh nhân trong 5 loại thuốc.
df = pd.read_csv('drug200.csv', header=0, delimiter=',', encoding='utf-8')
df.head(4)

# %%
# fig, ax = plt.subplots(figsize=(10, 3)) 
# ax.axis('off')  
# ax.axis('tight') 
# table = ax.table(
#     cellText=df.head(10).values,
#     colLabels=df.columns,
#     loc='center',
#     cellLoc='center'
# )
# table.auto_set_font_size(False)
# table.set_fontsize(10)
# table.scale(1.2, 1.2) 
# plt.savefig('dataframe_table.png', bbox_inches='tight', dpi=300)

# %% [markdown]
# Dự đoán thuốc thông qua chỉ số của bệnh nhân, sử dụng mô hình cây quyết định

# %%
# Kích thước tập dữ liệu
df.shape, df.columns

# %% [markdown]
# Khai phá -> Làm sạch -> Chuẩn hóa dữ liệu

# %%
df.info()

# %%
df.columns

# %%
# Các thang do lường của đặc trưng Cholesterol
df['Drug'].unique(), df['Cholesterol'].unique(), df['Sex'].unique(), df['BP'].unique()

# %%
# Kiểm tra dữ liệu bị thiếu
df.isna().sum()

# %%
# Trực quan hóa dữ liệu thiếu
plt.figure(figsize=(7,4))
sns.heatmap(df.isna().transpose(), 
            cmap='YlGnBu', 
            cbar_kws={'label': 'Missing Data'})
plt.title('Heatmap of Missing Data')
# plt.savefig('missing_data_heatmap.png')
plt.show()

# %%
# Kiểm tra tính hợp lệ các dữ liệu số
def validate_numerical_data(df):
    # Check if 'Age' and 'Na_to_K' columns exist
    required_columns = ['Age', 'Na_to_K']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the DataFrame.")
    # Check if the values in 'Age' and 'Na_to_K' are numerical and greater than 0
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numerical.")
        if (df[col] <= 0).any():
            raise ValueError(f"Column '{col}' must contain values greater than 0.")
    return True

# %%
val = validate_numerical_data(df)
if(val):
    print("Dữ liệu số đã hợp lệ.")
else:
    print("Dữ liệu số không hợp lệ.")

# %%
# Kiểm tra giá trị trùng lặp
df.duplicated().sum()

# %% [markdown]
# Trực quan hóa qua biểu đồ pie, bar, box-plot, line

# %%
# Đổi tên cột và dữ liệu định tính trong cột
df.rename(columns={ 
    'Na_to_K': 'Sodium_to_Potassium', 
    'BP': 'Blood_Pressure'
}, inplace=True) 
df['Sex'] = df['Sex'].replace({'M': 'Male', 'F': 'Female'})
# Làm tròn giá trị trong cột Sodium_to_Potassium
df['Sodium_to_Potassium'] = df['Sodium_to_Potassium'].round(0).astype(int)
df.head(4)

# %%
# Ma trận tương quan của Age, Sodium_to_Potassium
corr_age_na_to_k = df[['Age', 'Sodium_to_Potassium']].corr()
corr_age_na_to_k
# => Nhận xét: Không có tương quan giữa Age và Sodium_to_Potassium

# %%
# Ma trận hiệp phương sai giữa Age và Sodium_to_Potassium
cov_age_na_to_k =np.cov(df['Age'], df['Sodium_to_Potassium'])
cov_age_na_to_k

# %%
plt.figure(figsize=(8, 6))  # Đặt kích thước cho heatmap
sns.heatmap(corr_age_na_to_k, annot=True, fmt='.2f', cmap='YlGnBu')  
plt.title("Correlation Matrix", fontsize=16)
# plt.savefig("corr_matrix.png")
plt.show()


# %%
plt.figure(figsize=(8, 6))  # Đặt kích thước cho heatmap
sns.heatmap(cov_age_na_to_k, annot=True, fmt='.2f', cmap='YlGnBu')  
plt.title("Covariance Matrix", fontsize=16)
# plt.savefig("./result-images/cov_matrix.png")
plt.show()

# %%
# Đưa ra trung bình, trung vị, độ lệch chuẩn, khoảng phân vị của cột Age và Sodium_to_Potassium
df[['Age', 'Sodium_to_Potassium']].describe(include='all').round(2)

# %%
# Trực quan hóa pie chart về Cholesterol, Sex, Blood_Pressure và Drug
def plot_pie_chart(ax, column_name, df):
    gb = df.groupby([column_name])[column_name].agg(['count'])
    labels = gb.index 
    data = list(gb['count']) 
    colors = sns.color_palette('pastel') 
    ax.pie(data, labels=labels, colors=colors, 
           autopct='%1.1f%%', shadow=True)
    ax.set_title('Pie chart of ' + column_name)

# %%
# plt.figure(figsize=(4,1))
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
plot_pie_chart(axs[0, 0], 'Cholesterol', df)
plot_pie_chart(axs[0, 1], 'Sex', df)
plot_pie_chart(axs[1, 0], 'Blood_Pressure', df)
plot_pie_chart(axs[1, 1], 'Drug', df)
plt.tight_layout()
plt.show()

# %%
# Create plot
def create_plot(ax, x, data, plot_type='count', y=None, palette='Set2'):
    if plot_type == 'count':
        sns.countplot(x=x, hue=x, data=data, palette=palette, ax=ax)
    elif plot_type == 'bar':
        sns.barplot(x=x, hue=x, y=y, data=data, palette=palette, ax=ax)
    ax.set_title(f'Plot of {x}' if plot_type == 'count' else f'Bar plot of {x} and {y}')
    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height())}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='baseline',
            fontsize=10, color='black',
            xytext=(0, 1),
            textcoords='offset points')

# %%
# Sử dụng 6 vị trí: 2 hàng x 3 cột
plt.figure(figsize=(12, 10)) 
age_counts = df['Age'].count()
plot_configs = [
    {'x': 'Sex', 'data': df, 'plot_type': 'count'},
    {'x': 'Blood_Pressure', 'data': df, 'plot_type': 'count'},
    {'x': 'Cholesterol', 'data': df, 'plot_type': 'count'},
    {'x': 'Drug', 'data': df, 'plot_type': 'count'},
]
for i, config in enumerate(plot_configs):
    # Dùng 2 hàng, 3 cột
    ax = plt.subplot(2, 2, i + 1)
    create_plot(ax, **config)
plt.tight_layout()
plt.title('Categorical Plots')
# plt.savefig('categorical_plots.png')
plt.show()

# %%
# Mô tả tuổi theo nhóm giới tính
sns.displot(df, x='Age', hue='Sex', kind='kde')
plt.show()

# %%
# Mô tả chỉ số Na to K theo giới tính
sns.displot(df, x='Sodium_to_Potassium', hue='Sex', kind='kde')
plt.show()

# %%
# Phân phối và mức độ nhọn cột Age, Sodium_to_Potassium
def check_skewness(col):
    skewness = df[col].skew()
    if skewness > 1:
        return "Lệch phải mạnh (Highly right-skewed)"
    elif 0.5 < skewness <= 1:
        return "Lệch phải vừa phải (Moderately right-skewed)"
    elif -0.5 <= skewness <= 0.5:
        return "Gần đối xứng (Approximately symmetric)"
    elif -1 <= skewness < -0.5:   
        return "Lệch trái vừa phải (Moderately left-skewed)"
    else:  
        return "Lệch trái mạnh (Highly left-skewed)"
def check_kurtosis(col):
    kurt_value = df[col].kurtosis()
    if kurt_value > 0.1:
        return "Leptokurtic: đỉnh nhọn hơn và đuôi nặng."
    elif kurt_value < -0.1:
        return "Platykurtic: đỉnh phẳng hơn và đuôi nhẹ."
    else:
        return "Mesokurtic: đỉnh đều tạo phân phối chuẩn."
# if kurt_value > 0.1: # Sử dụng ngưỡng lớn hơn 0 một chút để có tính thực tế hơn
#     phan_loai = "Leptokurtic"
#     mo_ta = "Đỉnh nhọn hơn và đuôi nặng hơn (Kurtosis > 0)."
# elif kurt_value < -0.1: # Sử dụng ngưỡng nhỏ hơn 0 một chút
#     phan_loai = "Platykurtic"
#     mo_ta = "Đỉnh phẳng hơn và đuôi nhẹ hơn (Kurtosis < 0)."
# else:
#     phan_loai = "Mesokurtic"

# %%
print('Skewness Age:',check_skewness('Age'))
print('Skewness Sodium_to_Potassium:',check_skewness('Sodium_to_Potassium'))

# %%
print('Kurtosis Age:',check_kurtosis('Age'))
print('Kurtosis Sodium_to_Potassium:',check_kurtosis('Sodium_to_Potassium'))

# %%
sns.displot(data = df[['Age', 'Sodium_to_Potassium']], kind='kde')
# plt.savefig('kde_age_na_to_k.png')
plt.show()

# %% [markdown]
# Tính toán các đặc trưng cơ bản

# %%
# Tính chỉ số Gini cho cột Drug
def gini_index(list_values):
    arr = np.array(list_values)
    return 1 - np.sum((arr/np.sum(arr))**2)
gini_drug = gini_index(df['Drug'].value_counts())
print('Gini index of Drug:', gini_drug)
# Gini = 0.69405 => Nhận xét: Chỉ số Gini cho thấy sự đa dạng trong việc phân loại thuốc.

# %%
# Tính toán Entropy
def calculate_entropy(data, label_column):
    total_count = len(data)
    class_counts = data[label_column].value_counts()
    probabilities = class_counts / total_count
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy
# Gọi hàm để tính entropy
entropy_value = calculate_entropy(df, 'Drug')
print(f'Entropy of the dataset is: {entropy_value}')
# Entropy khoảng 1.9688 cho thấy tập dữ liệu có ít nhất 4 lớp (vì log₂(4) = 2).

# %% [markdown]
# # Xây dựng mô hình phân loại cây quyết định

# %%
# Chọn đặc trưng tập đầu vào và tập đầu ra
X = df.iloc[:,:-1] 
y = df.iloc[:, -1]

# %%
X

# %%
y

# %%
# One Hot Encoder cho cột 'Sex'
one_hot_encoder_sex = OneHotEncoder(sparse_output=False) 
# drop='first' để tránh multicollinearity (tùy chọn)
one_hot_encoded = one_hot_encoder_sex.fit_transform(X[['Sex']])  
# Dùng fit_transform
one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder_sex.get_feature_names_out(['Sex']), index=X.index)
X = pd.concat([X, one_hot_df], axis=1)
X = X.drop('Sex', axis=1)  # Drop cột gốc

# Label Encoder cho Blood_Pressure và Cholesterol
labelled_encoder_blood_pressure = LabelEncoder()
labelled_encoder_cholesterol = LabelEncoder()

X['Blood_Pressure'] = labelled_encoder_blood_pressure.fit_transform(X['Blood_Pressure'])
X['Cholesterol'] = labelled_encoder_cholesterol.fit_transform(X['Cholesterol'])

X['Sex_Female'] = X['Sex_Female'].round(0).astype(int)
X['Sex_Male'] = X['Sex_Male'].round(0).astype(int)
# Show features
# data
X

# %%
# Có thể thay đổi tham số
test_train_size = 0.4
random_state_to_split = 42

# %%
# Tách 2 tập với tỉ lệ là 6/4 random state là 42
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_train_size, random_state=random_state_to_split)

# %%
# Sử dụng Grid Search CV và K-Fold để tìm tham số tốt nhất cho mô hình Decision Tree
dtc = DecisionTreeClassifier(criterion='entropy')
# Determine fold as 5 and scoring accuracy
score_cross_val = cross_val_score(dtc, X_train, y_train, cv=5, scoring='accuracy').mean()
print('Performance of Decision Tree using Cross Validation:', score_cross_val)

# %%
# Đưa ra các tham số cây quyết định tốt nhất
dtc = DecisionTreeClassifier()
param_grid = {
    'max_depth': [3, 5, 10], 
    'min_samples_split': [2, 5, 10], 
    'criterion': ['gini', 'entropy'], 
    'max_leaf_nodes': range (5,10)
}
# 5-fold cross-validation, chọn fold là 5
grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# %%
# Tham số tốt nhất và độ chính xác tương ứng
print('Best Hyperparameters:', grid_search.best_params_)
print('Best Accuracy:', grid_search.best_score_)

# %%
max_dept_best_params = grid_search.best_params_['max_depth']
criterion_best_params = grid_search.best_params_['criterion']
max_leaf_nodes_best_params = grid_search.best_params_['max_leaf_nodes']
min_samples_leaf_best_params = grid_search.best_params_['min_samples_split']

# %%
# max_dept: độ sâu tối đa
# criterion: hàm đo độ tinh khiết để chọn nhánh
# max_leaf_nodes: số lượng lá tối đa trong cây
# min_samples_leaf: Số mẫu tối thiểu ở lá

# %%
# Tạo cây quyết định có chiều sâu đối đa là 6, node trái tối đa là 10
# dtc = DecisionTreeClassifier(criterion='entropy', max_depth=6, max_leaf_nodes=10)

dtc = DecisionTreeClassifier(max_depth=max_dept_best_params, criterion=criterion_best_params, 
                             max_leaf_nodes=max_leaf_nodes_best_params, 
                             min_samples_leaf=min_samples_leaf_best_params)
dtc.fit(X_train, y_train)

# %%
score_train = dtc.score(X_train, y_train)
score_train_dtc = score_train
score_test = dtc.score(X_test, y_test)
print(f'Accuracy on Training set: {score_train:4f}')
print(f'Accuracy on Testing set: {score_test:4f}')

# %%
# Phân loại lớp
dtc.classes_

# %%
# Trực quan cây quyết định
feature_cols = X_train.columns
plt.figure(figsize=(15,10))
plot_tree(dtc, class_names=dtc.classes_, feature_names=feature_cols, fontsize=12, filled=True)
# plt.savefig('decision_tree.png')
plt.show()

# %%
# Predict y test predict
# Dự đoán tập y test
y_train_pred = dtc.predict(X_train)
y_test_pred = dtc.predict(X_test)
y_test_pred_proba = dtc.predict_proba(X_test)

# %% [markdown]
# # Đánh giá mô hình

# %%
# Accuracy score, how ofter is the classifier correct ?
acc_score = accuracy_score(y_test, y_test_pred)
loss_score = log_loss(y_test, y_test_pred_proba)
accuracy_score_dtc = acc_score
loss_score_dtc = loss_score

print("Accuracy score:", 100*acc_score)
print("Loss score:", round(100 * loss_score, 2))

# %%
# Ma trận nhầm lẫn (Confusion Matrix)
cf_matrix = confusion_matrix(y_test, y_test_pred)
cf_matrix

# %%
def plot_confusion_matrix(cf_matrix):
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=dtc.classes_, yticklabels=dtc.classes_)
    plt.ylabel("Prediction", fontsize=12)
    plt.xlabel("Actual", fontsize=12)
    plt.title("Confusion Matrix", fontsize=16)
    plt.show()

# %%
# Plot the confusion matrix
plot_confusion_matrix(cf_matrix)

# %%
precision = precision_score(y_test, y_test_pred, average=None)
micro_precision = precision_score(y_test, y_test_pred, average='micro')
macro_precision = precision_score(y_test, y_test_pred, average='macro')
recall = recall_score(y_test, y_test_pred, average=None)
f1 = f1_score(y_test, y_test_pred, average=None)
report_classification = classification_report(y_test, y_test_pred)

# %%
print('Precision score:', precision)
print('Micro precision score:', micro_precision)
print('Macro precision score:', macro_precision)
print('Recall score:', recall)
print('F1 score:', f1)
print('Confusion Matrix:\n',cf_matrix)
print('Classification report:\n', report_classification)

# %%
# Prediction
age = 32
blood = 'HIGH'
blood_scaled = labelled_encoder_blood_pressure.transform([blood])

cholesterol = 'NORMAL'
cholesterol_scaled = labelled_encoder_blood_pressure.transform([cholesterol])

Sodium_to_Potassium = 13
# Số hóa ngược
sex = 'Female'
sex_female = 1 if sex == 'Female' else 0
sex_male   = 1 if sex == 'Male'   else 0

# %%
feature_sample = pd.DataFrame([[age, blood_scaled[0], 
                                cholesterol_scaled[0], 
                                Sodium_to_Potassium, 
                                sex_female, 
                                sex_male]], 
                              columns=X.columns)
y_sample_pred = dtc.predict(feature_sample)
y_sample_pred

# %%
X_train.columns

# %%
y_test

# %%
dfPredict_DTC = pd.DataFrame({
    'y_test': y_test,
    'y_test_predicted': y_test_pred,
})
dfPredict_DTC['Correct_Prediction'] = (y_test == y_test_pred).astype(int)
dfPredict_DTC

# %% [markdown]
# # So sánh với các mô hình Logistic Regression, K-Nearest Neighbors (KNN), Naive Bayes

# %% [markdown]
# Logistic

# %%
steps_to_fit_logistic=[('standardscaler',StandardScaler()),('logreg',LogisticRegression())]
pipe_logistic = Pipeline(steps_to_fit_logistic)
pipe_logistic.fit(X_train, y_train)

# %%
coefficients = pipe_logistic.named_steps['logreg'].coef_
intercept = pipe_logistic.named_steps['logreg'].intercept_
number_of_classes = pipe_logistic.named_steps['logreg'].classes_
print("Coefficients:", coefficients)
print("Intercept:", intercept)
print("Classes:", number_of_classes)

score_train = pipe_logistic.score(X_train, y_train)
score_train_logistic = score_train
score_test = pipe_logistic.score(X_test, y_test)
print(f'Accuracy on Training set: {score_train:4f}')
print(f'Accuracy on Testing set: {score_test:4f}')

y_train_pred = pipe_logistic.predict(X_train)
y_test_pred = pipe_logistic.predict(X_test)
y_test_pred_proba = pipe_logistic.predict_proba(X_test)

acc_score = accuracy_score(y_test, y_test_pred)
loss_score = log_loss(y_test, y_test_pred_proba)
accuracy_score_logistic = acc_score
loss_score_logistic = loss_score
print("Accuracy score:", 100*acc_score)
print("Loss score:", round(100 * loss_score, 2))

# Ma trận nhầm lẫn (Confusion Matrix)
cf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix for Logistic Regression:")
print(cf_matrix)
plot_confusion_matrix(cf_matrix)

precision = precision_score(y_test, y_test_pred, average=None)
micro_precision = precision_score(y_test, y_test_pred, average='micro')
macro_precision = precision_score(y_test, y_test_pred, average='macro')
recall = recall_score(y_test, y_test_pred, average=None)
f1 = f1_score(y_test, y_test_pred, average=None)
report_classification = classification_report(y_test, y_test_pred)

print('Precision score:', precision)
print('Micro precision score:', micro_precision)
print('Macro precision score:', macro_precision)
print('Recall score:', recall)
print('F1 score:', f1)
print('Classification report:\n', report_classification)

# %%
dfPredict_Logistic = pd.DataFrame({
    'y_test': y_test,
    'y_test_predicted': y_test_pred,
})
dfPredict_Logistic['Correct_Prediction'] = (y_test == y_test_pred).astype(int)
dfPredict_Logistic

# %% [markdown]
# KNN

# %%
# KNN Model
# Xác định số cụm đạt độ chính xác cao nhất
# Danh sách số lượng neighbors cần thử: từ 1 đến 50
range_first = 1
range_last = 50
neighbors = list(range(range_first, range_last))
accuracies = []
for k in neighbors:
    # Tạo và huấn luyện mô hình KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    # Dự đoán và tính độ chính xác
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
df_knn_results = pd.DataFrame({
    'n_neighbors': neighbors,
    'Accuracy': accuracies,
    'Accuracy (%)': [acc * 100 for acc in accuracies]
})
df_knn_results['Accuracy (%)'] = df_knn_results['Accuracy (%)'].round(4)
df_knn_results = df_knn_results.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
n_neighbors_best = df_knn_results.loc[0, 'n_neighbors']

a_index = list(range(range_first, range_last))
l_index = list(range(range_first, range_last))
a = pd.Series(dtype=float) 
l = pd.Series(dtype=float) 

df_knn_results.head(10)

# %%
# View plot result KNN Accuracy, Loss
plt.subplots(figsize=(20, 5))
for i in a_index:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    y_prediction = model.predict(X_test)
    # y_prediction_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_prediction)
    # loss = log_loss(y_test, y_prediction_proba)
    if accuracy is not None:
        a = pd.concat([a, pd.Series([accuracy])], ignore_index=True)
    # if loss is not None:
    #     l = pd.concat([l, pd.Series([loss])], ignore_index=True)
plt.plot(a_index, a, marker=".", color='blue', label = 'Accuracy')
# plt.plot(l_index, l, marker=".", color='red', label = 'Loss')
plt.xticks(a_index)
plt.xlabel('Index')
plt.ylabel('Accuracy')
plt.title('KNN Model Accuracy')
plt.show()

plt.subplots(figsize=(20, 5))
for i in a_index:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    y_prediction = model.predict(X_test)
    y_prediction_proba = model.predict_proba(X_test)
    loss = log_loss(y_test, y_prediction_proba)
    if loss is not None:
        l = pd.concat([l, pd.Series([loss])], ignore_index=True)
plt.plot(l_index, l, marker=".", color='red', label = 'Loss')
plt.xticks(a_index)
plt.xlabel('Index')
plt.ylabel('Loss')
plt.title('KNN Model Loss')
plt.show()

# %%
# Huấn luyện mô hình với k = 9
knnModel = KNeighborsClassifier(n_neighbors=n_neighbors_best)
knnModel.fit(X_train, y_train)

# %%
score_train = knnModel.score(X_train, y_train)
score_train_knn = score_train
score_test = knnModel.score(X_test, y_test)
print(f'Accuracy on Training set: {score_train:4f}')
print(f'Accuracy on Testing set: {score_test:4f}')

y_train_pred = knnModel.predict(X_train)
y_test_pred = knnModel.predict(X_test)
y_test_pred_proba = knnModel.predict_proba(X_test)

acc_score = accuracy_score(y_test, y_test_pred)
loss_score = log_loss(y_test, y_test_pred_proba)
accuracy_score_knn = acc_score
loss_score_knn = loss_score
print("Accuracy score:", 100*acc_score)
print("Loss score:", 100*loss_score)

# Ma trận nhầm lẫn (Confusion Matrix)
cf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix for KNN:")
print(cf_matrix)
plot_confusion_matrix(cf_matrix)

precision = precision_score(y_test, y_test_pred, average=None)
micro_precision = precision_score(y_test, y_test_pred, average='micro')
macro_precision = precision_score(y_test, y_test_pred, average='macro')
recall = recall_score(y_test, y_test_pred, average=None)
f1 = f1_score(y_test, y_test_pred, average=None)
report_classification = classification_report(y_test, y_test_pred)

print('Precision score:', precision)
print('Micro precision score:', micro_precision)
print('Macro precision score:', macro_precision)
print('Recall score:', recall)
print('F1 score:', f1)
print('Classification report:\n', report_classification)

# %%
dfPredict_KNN = pd.DataFrame({
    'y_test': y_test,
    'y_test_predicted': y_test_pred,
})
dfPredict_KNN['Correct_Prediction'] = (y_test == y_test_pred).astype(int)
dfPredict_KNN

# %% [markdown]
# Naive Bayes

# %%
# Naive Bayes Model
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)

# %%
score_train = gaussian.score(X_train, y_train)
score_train_nb = score_train
score_test = gaussian.score(X_test, y_test)
print(f'Accuracy on Training set: {score_train:4f}')
print(f'Accuracy on Testing set: {score_test:4f}')

y_train_pred = gaussian.predict(X_train)
y_test_pred = gaussian.predict(X_test)
y_test_pred_proba = gaussian.predict_proba(X_test)

acc_score = accuracy_score(y_test, y_test_pred)
loss_score = log_loss(y_test, y_test_pred_proba)
accuracy_score_nb = acc_score
loss_score_nb = loss_score
print("Accuracy score:", 100*acc_score)
print("Loss score:", 100*loss_score)

# Ma trận nhầm lẫn (Confusion Matrix)
cf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix for Naive Bayes:")
print(cf_matrix)
plot_confusion_matrix(cf_matrix)

precision = precision_score(y_test, y_test_pred, average=None)
micro_precision = precision_score(y_test, y_test_pred, average='micro')
macro_precision = precision_score(y_test, y_test_pred, average='macro')
recall = recall_score(y_test, y_test_pred, average=None)
f1 = f1_score(y_test, y_test_pred, average=None)
report_classification = classification_report(y_test, y_test_pred)

print('Precision score:', precision)
print('Micro precision score:', micro_precision)
print('Macro precision score:', macro_precision)
print('Recall score:', recall)
print('F1 score:', f1)
print('Classification report:\n', report_classification)

# %%
dfPredict_Bayes = pd.DataFrame({
    'y_test': y_test,
    'y_test_predicted': y_test_pred,
})
dfPredict_Bayes['Correct_Prediction'] = (y_test == y_test_pred).astype(int)
dfPredict_Bayes

# %% [markdown]
# # So sánh các mô hình trên

# %%
# So sánh mô hình cây quyết định tham số tốt, Logistic, KNN, Naive Bayes
result_report = {
    'Model': ['Decision Tree', 'Logistic Regression', 'KNN', 'Naive Bayes'],
    'Score': [score_train_dtc, score_train_logistic, score_train_knn, score_train_nb],
    'Accuracy score': [accuracy_score_dtc, accuracy_score_logistic, accuracy_score_knn, accuracy_score_nb],
    'Loss score': [loss_score_dtc, loss_score_logistic, loss_score_knn, loss_score_nb]
}
results = pd.DataFrame(result_report)
results_df = results.sort_values(by='Accuracy score', ascending=False)
results_df = results_df.reset_index(drop=True)
# results_df['Accuracy score percent'] = round(results_df['Accuracy score'] * 100, 2)
results_df

# %%
# results_df_rounded = results_df.copy() 
# results_df_rounded[['Score', 'Accuracy score', 'Loss score']] = results_df[['Score', 'Accuracy score', 'Loss score']].round(2)
# fig, ax = plt.subplots(figsize=(10, 3)) 
# ax.axis('off')  
# ax.axis('tight') 
# table = ax.table(
#     cellText=results_df_rounded.values,
#     colLabels=results_df_rounded.columns,
#     loc='center',
#     cellLoc='center'
# )
# table.auto_set_font_size(False)
# table.set_fontsize(10)
# table.scale(1.2, 1.2) 
# plt.savefig('comparison_table_models.png', bbox_inches='tight', dpi=300)

# %%
# Hàm hiển thị bar chart so sánh các mô hình
# def plot_bar_comparison(column, results_df):
#     plt.subplots(figsize=(7, 2))
#     ax = sns.barplot(x='Model', y=column, data=results_df)
#     labels = results_df[column] 
#     for i, v in enumerate(labels):
#         ax.text(i, v + 0.01, str(round(v*100,2)), horizontalalignment='center', size=12, color='black')
#     plt.title(column)
#     plt.show()

# %%
# plot_bar_comparison('Accuracy score', results_df)
# plot_bar_comparison('Loss score', results_df)

# %%
df_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Value')
plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x='Model', y='Value', hue='Metric', palette='viridis')
plt.title('Model Performance Metrics', fontsize=16)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(title='Metric')
plt.xticks(rotation=45)  
plt.tight_layout()  
# plt.savefig('barplot_model_comparison.png')
plt.show()

# %%
# So sánh các dự đoán chi tiết của các mô hình
print(f'Correct prediction rate Decision Tree: {dfPredict_DTC['Correct_Prediction'].sum()}/{len(dfPredict_DTC)}')
print(f'Correct prediction rate Logistic: {dfPredict_Logistic['Correct_Prediction'].sum()}/{len(dfPredict_Logistic)}')
print(f'Correct prediction rate KNN: {dfPredict_KNN['Correct_Prediction'].sum()}/{len(dfPredict_KNN)}')
print(f'Correct prediction rate Naive Bayes: {dfPredict_Bayes['Correct_Prediction'].sum()}/{len(dfPredict_Bayes)}')

# %%
dfPredict_DTC['Correct_Prediction_Text'] = dfPredict_DTC['Correct_Prediction'].map({
    1: 'Dự đoán đúng',
    0: 'Dự đoán sai'
})

dfPredict_Logistic['Correct_Prediction_Text'] = dfPredict_Logistic['Correct_Prediction'].map({
    1: 'Dự đoán đúng',
    0: 'Dự đoán sai'
})

dfPredict_KNN['Correct_Prediction_Text'] = dfPredict_KNN['Correct_Prediction'].map({
    1: 'Dự đoán đúng',
    0: 'Dự đoán sai'
})

dfPredict_Bayes['Correct_Prediction_Text'] = dfPredict_Bayes['Correct_Prediction'].map({
    1: 'Dự đoán đúng',
    0: 'Dự đoán sai'
})

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
plot_pie_chart(axs[0, 0], 'Correct_Prediction_Text', dfPredict_DTC)
plot_pie_chart(axs[0, 1], 'Correct_Prediction_Text', dfPredict_Logistic)
plot_pie_chart(axs[1, 0], 'Correct_Prediction_Text', dfPredict_KNN)
plot_pie_chart(axs[1, 1], 'Correct_Prediction_Text', dfPredict_Bayes)
plt.tight_layout()
plt.show()


