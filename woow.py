import pandas as pd
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# تحميل البيانات
file_path = r"C:\Users\Sec\Documents\wow\Electric Vehicle Population Data.csv"
data = pd.read_csv(file_path)

# واجهة المستخدم
st.title("التنبؤ بالمدى للسيارات الكهربائية ")
st.write("يتضمن هذا التطبيق خطوات تدريب النموذج والتنبؤ باستخدام أفضل الإعدادات.")

# معالجة القيم النصية
categorical_columns = ['Make', 'Model', 'Electric Vehicle Type', 'State', 'Electric Utility']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# إزالة القيم الناقصة
data = data.dropna(subset=['Electric Range', 'Base MSRP'])

# تحديد الميزات والهدف
features = ['Model Year', 'Make', 'Model', 'Electric Vehicle Type', 'Base MSRP', 'State', 'Electric Utility']
target = 'Electric Range'

X = data[features]
y = data[target]

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# عرض خطوات التدريب
st.header("خطوات تدريب النموذج")
st.write("1. تقسيم البيانات إلى مجموعة تدريب واختبار بنسبة 80% تدريب و20% اختبار.")
st.write("2. استخدام نموذج Random Forest مع ضبط المعلمات.")

# تدريب النموذج باستخدام GridSearchCV
param_grid = {
    'n_estimators': [300],  # أفضل عدد تم تحديده
    'max_depth': [10],      # أفضل عمق تم تحديده
    'min_samples_split': [10]  # أفضل قيمة للتقسيم
}
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='r2')
st.write("...جاري تدريب النموذج، قد يستغرق بعض الوقت...")
grid_search.fit(X_train, y_train)

# أفضل الإعدادات
best_params = grid_search.best_params_
st.success(f"تم العثور على أفضل الإعدادات: {best_params}")

# حفظ أفضل نموذج
best_model = grid_search.best_estimator_
model_path = r"C:\Users\Sec\Documents\wow\electric_range_model.pkl"
joblib.dump(best_model, model_path)
st.success("تم حفظ النموذج بنجاح!")

# التقييم
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# عرض نتائج التقييم
st.header("نتائج تقييم النموذج")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# عرض أهمية الميزات
st.header("أهمية الميزات")
importance = best_model.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

st.write("الميزات الأكثر تأثيرًا على التنبؤ:")
st.dataframe(importance_df)

# رسم أهمية الميزات
fig, ax = plt.subplots()
ax.barh(importance_df['Feature'], importance_df['Importance'])
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
ax.set_title(" feature_importances")
st.pyplot(fig)

# إضافة واجهة التنبؤ
st.header("تنبؤ جديد")
model_year = st.number_input("سنة التصنيع", min_value=2000, max_value=2025, step=1, value=2020)
make = st.selectbox("الشركة المصنعة", options=label_encoders['Make'].classes_)
base_msrp = st.number_input("السعر الأساسي ($)", min_value=1000, step=500, value=35000)
vehicle_type = st.selectbox("نوع السيارة", options=label_encoders['Electric Vehicle Type'].classes_)
state = st.selectbox("الولاية", options=label_encoders['State'].classes_)
electric_utility = st.selectbox("مزود الطاقة", options=label_encoders['Electric Utility'].classes_)

if st.button("احصل على التنبؤ"):
    # تجهيز البيانات
    user_data = pd.DataFrame(columns=X.columns)  # إنشاء DataFrame يحتوي على نفس الأعمدة المستخدمة في التدريب
    user_data.loc[0] = [
        model_year,
        label_encoders['Make'].transform([make])[0],
        0,  # قيمة افتراضية للعمود Model
        label_encoders['Electric Vehicle Type'].transform([vehicle_type])[0],
        base_msrp,
        label_encoders['State'].transform([state])[0],
        label_encoders['Electric Utility'].transform([electric_utility])[0]
    ]

    # التنبؤ
    prediction = best_model.predict(user_data)
    st.success(f"التنبؤ : {prediction[0]:.2f} ميل")
